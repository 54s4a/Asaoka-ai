# app/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np, json, os, yaml, re

app = FastAPI(title="Asaoka AI")

# ---- 起動時にインデックスをロード ----
INDEX = None
EMBS = None
INVARIANT = ""

def load_index():
    global INDEX, EMBS
    with open("rag/vector_store/index.json", encoding="utf-8") as f:
        data = json.load(f)
    INDEX = data["items"]
    EMBS = np.array(data["embeddings"], dtype=np.float32)

def load_invariant():
    global INVARIANT
    path = "prompts/system/00_invariants.yaml"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        # 短く詰めた system 文（不足ならそのまま文字列化）
        INVARIANT = (
            "あなたは仲介AIです。目的は衝突の冷却と合意形成です。"
            "出力は【核】【中立】【実務】【一体化まとめ】【次の一手】の順。"
            "敬体。Yes/No詰めの質問禁止。『』/「」の使い分けを守る。"
        )
    else:
        INVARIANT = "あなたは仲介AIです。出力フォーマットと禁則を守ってください。"

def cheap_embed(text: str, dim: int = 256) -> np.ndarray:
    # build_index.py と同じ手法（簡易ベクトル）
    import hashlib
    v = np.zeros(dim, dtype=np.float32)
    for ch in text:
        h = int(hashlib.md5(ch.encode('utf-8')).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v) or 1.0
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

@app.on_event("startup")
def _startup():
    load_index()
    load_invariant()

@app.get("/")
def root():
    return {"status": "ok", "kb_size": len(INDEX) if INDEX else 0}

# --------- /search ---------
@app.get("/search")
def search(q: str = Query(...), k: int = 12):
    qv = cheap_embed(q)
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1)+1e-9)
    ids = np.argsort(-sims)[:k].tolist()
    results = [{"score": float(sims[i]), **INDEX[i]} for i in ids]
    return {"query": q, "top_k": k, "results": results}

# --------- /answer ---------
class Ask(BaseModel):
    query: str
    k: int = 5

def format_answer(query: str, hits):
    # まずは骨組みの合成（OpenAI未使用のローカル合成）
    facts = "・" + "\n・".join([re.sub(r"\s+", " ", h["body"])[:120] + "…" for h in hits])
    return {
        "answer": f"""【核】
ご相談は「{query}」ですね。まず関連する前提を整理し、合意可能域を可視化いたします。

【中立】
{facts}

【実務】
・合意文（下書き）：（ここに状況に合わせた一文を置きます）
・選択式質問（Yes/No禁止）：（前提の掘り下げに使うオープン質問）
・禁句ガード：圧力表現や人格攻撃を避ける。

【一体化まとめ】
双方の利得と可逆性（後から取り消せる安全策）を確認します。

【次の一手】
24〜48時間でできる小さな行動案を1つだけ選びましょう。""",
        "used": [{"path": h["path"], "title": h["title"]} for h in hits],
        "invariant": INVARIANT
    }

@app.post("/answer")
def answer(payload: Ask):
    qv = cheap_embed(payload.query)
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1)+1e-9)
    ids = np.argsort(-sims)[: payload.k].tolist()
    hits = [INDEX[i] for i in ids]
    return format_answer(payload.query, hits)

from fastapi import FastAPI, Request
import os
import requests

app = FastAPI()

# ✅ 環境変数（RenderのEnvironmentで登録済みの2つを呼び出す）
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

# 🔹 LINE → AsaokaAI → LINE の接続ルート
@app.post("/line/webhook")
async def line_webhook(request: Request):
    body = await request.json()
    events = body.get("events", [])

    for event in events:
        if event.get("type") == "message":
            user_msg = event["message"]["text"]
            reply_token = event["replyToken"]

            # AsaokaAIの /answer エンドポイントへ問い合わせ
            res = requests.post(
                "https://asaoka-ai.onrender.com/answer",
                json={"question": user_msg}
            )

            # /answer からの返答を取得
            try:
                answer_text = res.json().get("answer", "処理エラーが発生しました。")
            except:
                answer_text = "AIの応答を取得できませんでした。"

            # LINEへ返信
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
            }
            data = {
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": answer_text}]
            }
            requests.post(LINE_REPLY_URL, headers=headers, json=data)

    return "OK"
