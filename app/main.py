# app/main.py
from fastapi import FastAPI, Query, Request, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import json, os, yaml, re
import hashlib
import requests
from typing import List, Dict, Any, Optional

app = FastAPI(title="Asaoka AI")

# =============
# 起動時にインデックスをロード
# =============
INDEX = None          # type: Optional[List[Dict[str, Any]]]
EMBS = None           # type: Optional[np.ndarray]
INVARIANT = ""        # type: str

def load_index():
    """ベクターストアと原文インデックスをロード"""
    global INDEX, EMBS
    path = "rag/vector_store/index.json"
    if not os.path.exists(path):
        INDEX = []
        EMBS = np.zeros((0, 256), dtype=np.float32)
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    INDEX = data.get("items", [])
    EMBS = np.array(data.get("embeddings", []), dtype=np.float32)

def load_invariant():
    """不変コアの読み込み（無ければ既定文言）"""
    global INVARIANT
    path = "prompts/system/00_invariants.yaml"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        INVARIANT = y.get("system", "あなたは仲介AIです。出力フォーマットと禁則を守ってください。")
    else:
        INVARIANT = "あなたは仲介AIです。出力フォーマットと禁則を守ってください。"

def cheap_embed(text: str, dim: int = 256) -> np.ndarray:
    """簡易埋め込み（決定論的・軽量）。本番は高性能埋め込みへ差し替え可能。"""
    v = np.zeros(dim, dtype=np.float32)
    for ch in text:
        h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v) or 1.0
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9) * (np.linalg.norm(b)+1e-9)))

@app.on_event("startup")
def _startup():
    load_index()
    load_invariant()

@app.get("/")
def root():
    return {"status": "ok", "kb_size": len(INDEX) if INDEX else 0}

# =============
# /search
# =============
@app.get("/search")
def search(q: str = Query(...), k: int = 12):
    qv = cheap_embed(q)
    if EMBS is None or EMBS.size == 0:
        return {"query": q, "top_k": k, "results": []}
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1)+1e-9)
    ids = np.argsort(-sims)[:k].tolist()
    results = [{"score": float(sims[i]), **INDEX[i]} for i in ids]
    return {"query": q, "top_k": k, "results": results}

# =============
# /answer
# =============
class Ask(BaseModel):
    query: str
    k: int = 12

def format_answer(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """既存のテンプレに合わせて合成。必要ならここを調整。"""
    facts = "・" + "\n・".join([re.sub(r'\s+', ' ', h.get("body",""))[:120] + "…" for h in hits])
    text = f"""【核】
ご相談は「{query}」です。まず事実に寄る前提整理を優先し、合意可能域を可視化いたします。

【中立】
{facts}

【実務】
- 選択式質問（Yes/Noは禁止）：（前提の掘り下げに使うオープン質問）
- 禁句ガード：圧力表現や人格攻撃を避ける。

【一体化まとめ】
双方が納得可能性（後から取り消せる安全策）を確認します。

【次の一手】
24〜48時間でできる小さな行動案を1つだけ選びましょう。
"""
    return {"answer": text, "used": [{"path": h.get("path"), "title": h.get("title")} for h in hits], "invariant": INVARIANT}

def make_answer_from_query(query: str, k: int = 12) -> Dict[str, Any]:
    """内部関数：HTTP を経由せずに RAG で回答を生成"""
    if EMBS is None or EMBS.size == 0 or not INDEX:
        return {"answer": "知識ベースが未ロードのため、今は一般指針のみで回答いたします。"}
    qv = cheap_embed(query)
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1)+1e-9)
    ids = np.argsort(-sims)[:k].tolist()
    hits = [INDEX[i] for i in ids]
    return format_answer(query, hits)

@app.post("/answer")
def answer(payload: Ask):
    return make_answer_from_query(payload.query, payload.k)

# =============
# 運用補助エンドポイント
# =============
VERSION = "line-integration-bg-v1"

@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "kb_size": len(INDEX) if INDEX else 0}

# =============
# LINE Webhook（BackgroundTasksで即200）
# =============
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

def reply_job(reply_token: str, user_msg: str):
    """バックグラウンドで /answer 相当を実行し、LINE に返信"""
    # 安全側に倒す（例外握り潰し＋フォールバック）
    try:
        ans = make_answer_from_query(user_msg, k=12)
        answer_text = ans.get("answer") or "応答の生成に失敗しました。"
    except Exception:
        answer_text = "ただいま応答が混み合っています。少し時間をおいて再送してください。"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": answer_text}]
    }
    try:
        requests.post(LINE_REPLY_URL, headers=headers, json=payload, timeout=5)
    except Exception:
        # 返信に失敗してもWebhook処理自体は完了させる
        pass

@app.post("/line/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    """LINE からの Webhook を受け取り、即 200 を返す。返信は BG タスクで送信。"""
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}

    events = body.get("events", [])
    for event in events:
        if event.get("type") == "message":
            msg = event.get("message", {}) or {}
            if msg.get("type") == "text":
                user_msg = msg.get("text", "")
            else:
                user_msg = "テキストメッセージで送信してください。"
            reply_token = event.get("replyToken")
            if reply_token:
                background_tasks.add_task(reply_job, reply_token, user_msg)

    # 即時応答（LINE 側のタイムアウト回避）
    return {"ok": True}
