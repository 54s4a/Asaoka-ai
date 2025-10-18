AsaokaAI – 仲介AIシステム
🧭 概要

AsaokaAI は、「合意形成」を目的とした仲介型AIです。
思想をそのままAIに埋め込むのではなく、**不変コア（人格）とRAG（知識文庫）**の二層構造で構築しています。

この仕組みにより、AIは感情的な反応をせず、常に中立かつ可逆的に問題を整理し、双方の合意を導き出すことができます。

🧱 システム構成
asaoka-ai/
├─ app/
│   ├ main.py               # Render用API
│   ├ requirements.txt
│   └ render.yaml
│
├─ prompts/system/
│   ├ 00_invariants.yaml    # 不変コア（人格・ルール）
│   ├ core-002.yaml         # CLAIM-002対応ルール
│   ├ core-003.yaml         # CLAIM-003対応ルール
│   └ core-common.yaml      # 全CLAIM共通制御
│
├─ kb/                      # チャンク文庫（RAG）
│   ├ workplace_feedback_CLAIM-002_0006.md
│   ├ family_distance_CLAIM-003_0012.md
│   └ ...
│
├─ tools/
│   ├ extract.py            # 事実・意図・条件の抽出
│   ├ guardrails.py         # 禁句・圧力検出
│   ├ meter.py              # 合意メーター（0〜100）
│   └ style_polish.py       # 敬体・『』『』整形
│
├─ tests/gold/
│   └ sample_case.json      # 回帰テスト用
│
└─ README.md                # このファイル

🧩 二層構造の考え方
層	役割	内容
不変コア（Invariant）	AIの人格・性格・出力契約	「敬体で話す」「Yes/No詰め禁止」「やり返し非提示」など絶対ルール
RAG（チャンク文庫）	思想・具体例・ケース	本や原稿を500〜800字ごとに分割してタグ付き保存。AIが必要時のみ検索
🧠 チャンク設計ルール

分割単位：1つの問い・考えごとに500〜800文字

命名規則：kb/<theme>/<slug>_CLAIM-XXX_YYYY.md

CLAIM体系：思想の幹（XXX）＋枝番（YYYY）

---
title: フィードバックの口調は事実→意図→合意の順で整える
theme: workplace
phase: 前提抽出
guard: 圧力回避
style: 敬体/『』『』/「」準拠
claim: CLAIM-002_0006
date: 2025-10-19
---
本文（500〜800字）…

🔸 タグの意味
項目	説明
theme	テーマ（恋愛・職場・親子など）
phase	段階（前提抽出・仮説提示・合意化など）
guard	注意点（禁句・圧力回避など）
style	文体ルール（敬体・『』『』・「」）
claim	思想の幹＋枝番号
date	作成・更新日
🔍 検索・参照構造（RAG）

AIは質問を受けると、以下の流れで情報を参照します：

Top-k 検索（上位12件）
10万件以上の中から関連度の高い上位12件を抽出。

再ランク（必要時のみ）
関連スコアが拮抗している場合のみ順位を付け直す。

抽出（3〜5件）
実際に使うのは上位3〜5件。

キャッシュ
同テーマが続く場合は前回結果を使い回す。

ログ
検索時間や生成時間を記録して改善に活用。

⚙️ API構成（Render × FastAPI）
/kb-index

kb/ 内のMarkdown一覧をJSONで返す（トークン認証あり）

ChatGPTがここを見て「既存チャンク」を把握し、重複を避けて新規を提案します。

/kb-diff

ChatGPTから「すでに持っているファイル名・ハッシュ値」を受け取り、
まだ存在しないチャンクだけを返す。

/healthz

Renderの稼働確認用。

🔐 セキュリティ運用

トークンはRenderの環境変数で管理（コード内に書かない）

トークンはBearerヘッダーで渡す

公開URLやSNSでトークンを共有しない

Renderは非公開APIとして動作（検索エンジン非表示）

🧭 不変コア（Invariant）構成
ファイル	役割
00_invariants.yaml	共通コア（人格・敬体・出力フォーマット）
core-002.yaml	CLAIM-002対応ルール（感情と構造理解）
core-003.yaml	CLAIM-003対応ルール（期待と自立）
00_invariants.yaml（例）
role: 仲介AI
purpose:
  - 衝突の冷却
  - 合意形成と可逆性の確保
contract:
  output_order: ["【核】","【中立】","【実務】","【一体化まとめ】","【次の一手】"]
  tone: 敬体・断定過剰回避・非煽動
  quotes: 『』『』/「」厳守
  bans:
    - Yes/No詰め
    - 人格攻撃
    - “やり返し”の原則提示（限定条件時のみ明示）
tools:
  - extract_facts_intent_conditions
  - agreement_meter
  - guardrails
  - style_polish
links:
  - CLAIM-002
  - CLAIM-003

🧪 テスト・回帰

tests/gold/ に理想出力を保存し、禁句・出力欠落・合意メーター閾値などを自動チェックします。

例：

{
  "input": "上司の注意が感情的で、話し合いになりません。",
  "expect_format": ["【核】","【中立】","【実務】","【一体化まとめ】","【次の一手】"],
  "bans": ["やり返し","Yes/Noで答えて"],
  "min_meter": 70
}

📈 パフォーマンス指標
処理	平均時間
近似検索（Top-12）	0.05〜0.2秒
再ランク（必要時）	+0.05〜0.12秒
本文取得	数ms〜数十ms
合計	0.1〜0.4秒（遅くても1秒以内）
🧭 運用のコツ

System（コア）を軽く、RAGを太く

思想更新はRAGの差し替えで完結

禁句・整形はツール関数で処理

回帰テストで壊れない構成を維持

✍️ 著者・思想提供

浅岡（Asaoka）
– 著書『あなたの人生のハイライトに、今日のあなたは居ますか？』
– AsaokaAI 企画・設計・執筆
