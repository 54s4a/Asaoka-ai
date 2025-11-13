# app/main.py
from fastapi import FastAPI, Query, Request, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import json, os, yaml, re
import hashlib
import requests
import hmac, base64  # ★ここを追加
from typing import List, Dict, Any, Optional
import datetime as dt

app = FastAPI(title="Asaoka AI")

# =====================
# 起動時にインデックスをロード
# =====================
INDEX: Optional[List[Dict[str, Any]]] = None
EMBS: Optional[np.ndarray] = None
INVARIANT: str = ""


def load_index():
    """ベクターストアと索引インデックスをロード"""
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
    """不変コアの読み込み（無ければ既定文）"""
    global INVARIANT
    path = "prompts/system/00_invariants.yaml"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        INVARIANT = y.get("system", "あなたは仲介AIです。出力フォーマットと禁則を守ってください。")
    else:
        INVARIANT = "あなたは仲介AIです。出力フォーマットと禁則を守ってください。"


def cheap_embed(text: str, dim: int = 256) -> np.ndarray:
    """簡易埋め込み（低速域・軽量）→ 本番は高性能埋め込みに差し替え可能。"""
    v = np.zeros(dim, dtype=np.float32)
    for ch in text:
        h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v) or 1.0
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


@app.on_event("startup")
def _startup():
    load_index()
    load_invariant()


@app.get("/")
def root():
    return {"status": "ok", "kb_size": len(INDEX) if INDEX else 0}


# ============
# /search
# ============
@app.get("/search")
def search(q: str = Query(...), k: int = 8):
    qv = cheap_embed(q)
    if EMBS is None or EMBS.size == 0:
        return {"query": q, "top_k": k, "results": []}
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1) + 1e-9)
    ids = np.argsort(-sims)[:k].tolist()
    results = [{
        "score": float(sims[i]),
        **{k2: INDEX[i].get(k2) for k2 in ("path", "title", "body")}
    } for i in ids]
    return {"query": q, "top_k": k, "results": results}


# ============
# /answer
# ============
class Ask(BaseModel):
    query: str
    k: int = 8  # 軽量化（初回遅延を抑制）


def format_answer(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """既存のテンプレに合わせて合成。必要なら更に調整。"""
    # hits から短い要約断片を生成（空ならプレースホルダ）
    if hits:
        facts_list = []
        for h in hits:
            body = (h.get("body") or "").replace("\n", " ")
            body = re.sub(r"\s+", " ", body)
            facts_list.append(body[:60] + "…")
        facts = "・" + "\n・".join(facts_list[:6])
    else:
        facts = "（知識ベースから関連情報を取得できませんでした。仮説整理のみ提示します。）"

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
    return {
        "answer": text,
        "used": [{"path": h.get("path"), "title": h.get("title")} for h in hits],
        "invariant": INVARIANT
    }


def make_answer_from_query(query: str, k: int = 8) -> Dict[str, Any]:
    """内部関数：HTTP を経由せずに RAG で回答を生成"""
    if EMBS is None or EMBS.size == 0 or not INDEX:
        # 知識ベース未ロード（起動直後など）のフォールバック
        return {"answer": "初期ロード中のため、今回は仮の整理のみで返信いたします。少し時間をおいて再送ください。"}

    qv = cheap_embed(query)
    sims = EMBS @ qv / (np.linalg.norm(EMBS, axis=1) + 1e-9)
    ids = np.argsort(-sims)[:k].tolist()
    hits = [INDEX[i] for i in ids]
    return format_answer(query, hits)


@app.post("/answer")
def answer(payload: Ask, request: Request):
    # 内部APIキーが設定されている場合だけチェック
    if INTERNAL_API_KEY:
        api_key = request.headers.get("X-API-Key")
        if api_key != INTERNAL_API_KEY:
            return {"ok": False, "reason": "forbidden"}

    return make_answer_from_query(payload.query, payload.k)


# =====================
# 運用確認用エンドポイント
# =====================
VERSION = "line-integration-bg-v1.1"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "kb_size": len(INDEX) if INDEX else 0,
        "now_ms": int(dt.datetime.now().timestamp() * 1000)
    }


# =====================
# LINE Webhook（BackgroundTasksで即200）
# =====================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")  # /answer 用の内部キー
def verify_line_signature(channel_secret: str, body_bytes: bytes, signature: str) -> bool:
    """LINE の X-Line-Signature を検証する"""
    if not channel_secret or not signature:
        return False
    mac = hmac.new(channel_secret.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)

# ── 4区分ルーティング用（追記） ─────────────────────────
IGNORE_PATTERNS = [
    "こんにちは","こんばんは","おはよう","やあ","はい","うん","なるほど",
    "ありがとう","了解","お疲れ","おつかれ","すごい","そうだね","そうですね","わかりました",
    "なる","うける","笑","ｗ","www","!","ok","okay","了解です","了解しました"
]
SOS_SHORTS = ["疲れた","しんどい","限界","つらい","無理","やめたい","きつい","もう無理"]
QUESTION_WORDS = ["どう","なぜ","何","どこ","いつ","誰","どんな","どう思う","なに","どれ"]
RELATION_WORDS = [
    "合わない","伝わら","価値観","気まず","上司","同僚","彼","彼女","親","誤解","衝突",
    "言い方","圧","モラハラ","パワハラ","仲介","調整","すり合わせ","折り合い"
]

def classify_intent(text: str) -> str:
    """
    4区分: "noise" | "light" | "chat" | "mediate"
    返信の“長さと深さ”だけを前段で切り替えます（既存のRAGは維持）。
    """
    if text is None:
        return "noise"
    t = text.strip()
    if not t:
        return "noise"
    # 記号・数字のみ
    if re.fullmatch(r"[ \t\n\r\W\d_]+", t):
        return "noise"
    # 2文字以下はノイズ
    if len(t) <= 2:
        return "noise"
    tl = t.lower()
    if any(kw in t or kw in tl for kw in IGNORE_PATTERNS):
        return "noise"
    if any(t.startswith(s) or s in t for s in SOS_SHORTS):
        return "light"
    if ("？" in t or "?" in t) or any(q in t for q in QUESTION_WORDS):
        return "chat"
    if any(k in t for k in RELATION_WORDS):
        return "mediate"
    return "chat"


def build_noise_reply(user_msg: str) -> str:
    # 10文字級の極短
    if "ありがとう" in user_msg:
        return "こちらこそ。大丈夫です。"
    if "おはよう" in user_msg:
        return "おはようございます。"
    return "どうも。大丈夫です。"

def build_light_reply(user_msg: str) -> str:
    # 共感＋確認のみ（長文禁止）
    if "限界" in user_msg:
        return "相当ご無理をされていますね。今は安全確保を最優先で大丈夫です。"
    return "無理をされているご様子ですね。必要でしたら一言だけ要点を教えてください。"

def build_chat_reply(user_msg: str) -> str:
    # 一般質問に2〜4文で簡潔回答（プレースホルダ）
    if "どう思う" in user_msg or "どうおもう" in user_msg:
        return "背景・目的・制約の三点で見方が変わります。各1行で共有いただければ、筋の通る選択肢に整理いたします。"
    if "なぜ" in user_msg:
        return "原因は『構造・運用・人』に分かれることが多いです。どの層か仮置きし、1点だけ検証しましょう。"
    return "要点を二つに絞ってお知らせください。前提確認のうえ簡潔にお答えいたします。"

def build_mediation_reply_short(user_msg: str) -> str:
    # 仲介テンプレ（短縮版。既存RAGは別途呼ぶ前提）
    return (
        "【核】『価値観／役割／裁量／人間関係／条件』から近いものを2つまで。\n"
        "【中立】直近2週間の影響（仕事／心身／退路）を各1行。\n"
        "【実務】1行ミスマッチ→15分面談打診→前後比較2ケース→体調セーフティ。"
    )


def reply_job(retry_token: str, user_msg: str):
    """バックグラウンドで最適レイヤーを選び、LINE に返信"""
    try:
        intent = classify_intent(user_msg)

        if intent == "noise":
            answer_text = build_noise_reply(user_msg)

        elif intent == "light":
            answer_text = build_light_reply(user_msg)

        elif intent == "chat":
            # Chat層は簡潔応答（必要なら独立APIに差し替え可）
            answer_text = build_chat_reply(user_msg)

        else:  # "mediate"
            # ★既存RAGを維持（/answer相当のロジック）
            ans = make_answer_from_query(user_msg, k=8)
            answer_text = ans.get("answer") or build_mediation_reply_short(user_msg)

    except Exception:
        answer_text = "ただいま処理が混み合っています。少し時間をおいて再送してください。"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "replyToken": retry_token,
        "messages": [{"type": "text", "text": answer_text[:4700]}]  # LINEの文字数上限対策
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
            # ログ用：LINEイベントの送信時刻と現在時刻を比較（リトライ判断用）
            evt_ms = event.get("timestamp")
            now_ms = int(dt.datetime.now().timestamp() * 1000)
            print(f"[LINE] event_ts={evt_ms} now_ms={now_ms} diff_ms={now_ms - (evt_ms or now_ms)}")
            if reply_token:
                background_tasks.add_task(reply_job, reply_token, user_msg)

    # 即時応答（LINE 側のタイムアウト回避）
    return {"ok": True}
