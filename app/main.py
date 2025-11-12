import os
import re
from typing import Any, Dict, List
import httpx
from fastapi import FastAPI, BackgroundTasks, Request

app = FastAPI(title="AsaokaAI Router (Chat/Mediate/Light/Noise)", version="1.1.0")

LINE_MESSAGING_API = "https://api.line.me/v2/bot/message/reply"
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

# ──────────────────────────────────────────────────────────────
# ルール：割合の初期値（参考値・コードには直接は使わない）
# 無視30%・軽受け15%・Chat30%・仲介25% を想定した分岐設計
# ──────────────────────────────────────────────────────────────

# 雑談・挨拶系（無視/短文応答）
IGNORE_PATTERNS: List[str] = [
    "こんにちは","こんばんは","おはよう","やあ","はい","うん","なるほど",
    "ありがとう","了解","お疲れ","おつかれ","すごい","そうだね","そうですね","わかりました",
    "なる","うける","笑","ｗ","www","!","ok","okay","了解です","了解しました"
]

# SOS短文（軽受け）：短く寄り添い＋要否確認だけ
SOS_SHORTS: List[str] = ["疲れた","しんどい","限界","つらい","無理","やめたい","きつい","もう無理"]

# 質問検知（Chat層）：疑問符または疑問語
QUESTION_WORDS: List[str] = ["どう","なぜ","何","どこ","いつ","誰","どんな","どう思う","なに","どれ"]

# 仲介トリガ（Mediate層）：関係・価値観・衝突
RELATION_WORDS: List[str] = [
    "合わない","伝わら","価値観","気まず","上司","同僚","彼","彼女","親","誤解","衝突",
    "言い方","圧","モラハラ","パワハラ","仲介","調整","すり合わせ","折り合い"
]


def classify_intent(text: str) -> str:
    """
    発話を4区分に分類する。
    戻り値: "noise" | "light" | "chat" | "mediate"
    """
    if text is None:
        return "noise"
    t = text.strip()
    if not t:
        return "noise"

    # 記号や数字だけ
    if re.fullmatch(r"[ \t\n\r\W\d_]+", t):
        return "noise"

    # 極短（2文字以下）
    if len(t) <= 2:
        return "noise"

    # 挨拶・相槌
    tl = t.lower()
    if any(kw in t or kw in tl for kw in IGNORE_PATTERNS):
        return "noise"

    # SOS短文（軽受け）：完全一致 or 含有（語尾助詞を許容）
    if any(t.startswith(s) or s in t for s in SOS_SHORTS):
        return "light"

    # 質問：疑問符 or 疑問語
    if ("？" in t or "?" in t) or any(q in t for q in QUESTION_WORDS):
        return "chat"

    # 関係・価値観・衝突語
    if any(k in t for k in RELATION_WORDS):
        return "mediate"

    # 既定：Chat層
    return "chat"


def build_noise_reply(user_msg: str) -> str:
    # 無視に近い軽い返答（10文字前後）
    if "ありがとう" in user_msg:
        return "こちらこそ。大丈夫です。"
    if "おはよう" in user_msg:
        return "おはようございます。"
    return "どうも。大丈夫です。"


def build_light_reply(user_msg: str) -> str:
    # SOS短文：共感＋確認のみ（長文にしない）
    if "限界" in user_msg:
        return "相当追い詰められているご様子ですね。今は安全確保を最優先にして大丈夫です。整理は後で構いません。"
    return "無理をされているご様子ですね。今すぐ整理が必要でしたら一言だけ要点を教えてください。"

async def chat_ai_answer(user_msg: str) -> str:
    """
    Chat層の簡易応答（2〜4文）。
    本番ではRAGなしの一般回答や、専用エンドポイント呼び出しに置き換え。
    """
    # ここはプレースホルダ。必要に応じて内部APIに差し替え可。
    if "どう思う" in user_msg or "どうおもう" in user_msg:
        return "状況により見方が変わります。背景・目的・制約の三点を一行ずつ共有いただければ、筋の通る選択肢に整理いたします。"
    if "なぜ" in user_msg:
        return "原因は『構造・運用・人』に分かれることが多いです。どの層で起きているか仮置きし、1つだけ確認しましょう。"
    return "要点を二つに絞ってお知らせください。前提を確認したうえで簡潔にお答えいたします。"

def build_mediation_reply(user_msg: str) -> str:
    # 仲介AI用の構造化テンプレ（前回提示の短縮版）
    return (
        "【核｜ズレの仮説】『価値観／役割／裁量／人間関係／条件』のうち近いものを2つまでお知らせください。\n"
        "【中立｜影響】直近2週間での影響（仕事／心身／退路）を一行ずつ。\n"
        "【実務｜48時間】1行ミスマッチ文→15分面談打診→前後比較2ケース→体調セーフティ。\n"
        "【返信フォーマット】1.種類 2.事例(事実/解釈/感情) 3.体調 4.面談可否"
    )


async def line_reply(reply_token: str, text: str) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[WARN] Missing LINE_CHANNEL_ACCESS_TOKEN. Simulated reply:", text[:200])
        return
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:4900]}],
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(LINE_MESSAGING_API, headers=headers, json=payload)
        if resp.status_code >= 300:
            print("[ERROR] LINE reply failed:", resp.status_code, resp.text)


async def route_and_build_reply(user_msg: str) -> str:
    intent = classify_intent(user_msg)
    if intent == "noise":
        return build_noise_reply(user_msg)
    if intent == "light":
        return build_light_reply(user_msg)
    if intent == "chat":
        return await chat_ai_answer(user_msg)
    # mediate
    return build_mediation_reply(user_msg)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": "AsaokaAI Router"}

@app.post("/line/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    events = body.get("events", [])
    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue
        user_msg = msg.get("text", "") or ""
        reply_token = ev.get("replyToken", "")

        # 同期で即返答（全層共通で短文）にして体感速度を上げる
        # ※必要ならHeavy処理をbackground_tasksに退避
        reply_text = await route_and_build_reply(user_msg)
        await line_reply(reply_token, reply_text)

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
