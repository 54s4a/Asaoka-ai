# rag/build_index.py
import os, json, glob, re, hashlib
import numpy as np

# --- 超軽量の文字ベース埋め込み（OpenAIを後で切替可） ---
def cheap_embed(text: str, dim: int = 256) -> np.ndarray:
    # 文字をハッシュして固定長ベクトル化（外部API不要、まずは動作確認用）
    v = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text):
        h = int(hashlib.md5(ch.encode('utf-8')).hexdigest(), 16)
        v[h % dim] += 1.0
    # 正規化
    n = np.linalg.norm(v) or 1.0
    return v / n

def parse_md(md: str):
    def pick(key):  # yaml風見出しを拾う（無くてもOK）
        m = re.search(rf'^{key}:\s*(.+)$', md, flags=re.M)
        return m.group(1).strip() if m else ""
    title = pick("title")
    phase = pick("phase")
    theme = pick("theme")
    guard = pick("guard")
    # body セクション抽出（summary/body/insight が無くても全文でOK）
    body_m = re.search(r'(?s)^body:\s*(.+?)(?:\n\w+:|$)', md, flags=re.M)
    body = body_m.group(1).strip() if body_m else md
    text_for_embed = " ".join([title, phase, theme, guard, body])[:3000]
    return {
        "title": title or "(no title)",
        "phase": phase, "theme": theme, "guard": guard,
        "body": body, "text": text_for_embed
    }

def main():
    kb_files = sorted(glob.glob("kb/*.md"))
    items, embs = [], []
    for path in kb_files:
        with open(path, encoding="utf-8") as f:
            md = f.read()
        meta = parse_md(md)
        meta["path"] = path.replace("\\", "/")
        items.append(meta)
        embs.append(cheap_embed(meta["text"]).tolist())

    os.makedirs("rag/vector_store", exist_ok=True)
    with open("rag/vector_store/index.json", "w", encoding="utf-8") as f:
        json.dump({"items": items, "embeddings": embs}, f, ensure_ascii=False)
    print(f"indexed {len(items)} files -> rag/vector_store/index.json")

if __name__ == "__main__":
    main()
