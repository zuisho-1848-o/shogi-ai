# explain/ — 盤面解説AI モジュール

現在の局面・候補手・評価値・定跡・格言を組み合わせて、
将棋の盤面を自然言語で解説するAIモジュール。

---

## 概要

```
SFEN + 候補手リスト
        ↓
context_builder   ← 盤面を人間が読めるテキストに変換
        ↓
knowledge         ← 定跡照合・格言マッチ・有名局面検出
        ↓
commentator       ← LLM呼び出し（Ollama / Claude API）
        ↓
解説テキスト（日本語 Markdown）
```

---

## ディレクトリ構成

```
explain/
  README.md              # このファイル
  DESIGN.md              # 詳細設計・データフロー
  PROGRESS.md            # 実装進捗・完了済み項目・次のステップ
  __init__.py
  context_builder.py     # SFEN → 構造化テキスト変換
  knowledge.py           # 定跡・格言・有名局面照合
  commentator.py         # LLMバックエンド抽象化 + 解説生成
  api.py                 # FastAPIルーター（/explain エンドポイント）
  data/
    proverbs.json        # 将棋格言DB
    joseki_map.json      # 主要定跡の代表SFEN・戦型パターン
    famous_positions.json  # 有名局面DB（SFEN + 棋戦名）
```

---

## LLMバックエンド

`commentator.py` は抽象クラスで切り替え可能。

| バックエンド | 設定値 | 備考 |
|------------|--------|------|
| Ollama（ローカル） | `ollama` | デフォルト。`llama3`, `gemma3`, `qwen2.5` 等 |
| Claude API | `claude` | `ANTHROPIC_API_KEY` 必須 |

環境変数または `/explain` リクエストのパラメータで切り替え。

```
LLM_BACKEND=ollama         # or "claude"
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
ANTHROPIC_API_KEY=sk-...   # claudeの場合のみ
```

---

## 解説レベル

| レベル | 対象 | 特徴 |
|--------|------|------|
| `beginner` | 初心者〜3級 | 駒の名前・基本用語から説明、平易な日本語 |
| `intermediate` | 初段前後 | 手筋名・定跡名を使う、形勢判断を含む |
| `advanced` | 三段以上 | 変化手順・深い読み・プロの視点で解説 |

リクエストパラメータ `level` で指定。デフォルトは `intermediate`。

---

## APIエンドポイント（予定）

```
POST /explain
  sfen: string               # 現在の局面（SFEN形式）
  candidates: CandidateMove[]  # 候補手リスト（既存/analyse の出力をそのまま渡せる）
  history_sfens: string[]    # 棋譜（任意）
  level: "beginner" | "intermediate" | "advanced"
  backend: "ollama" | "claude"  # 省略時は環境変数優先

レスポンス:
  commentary: string         # 解説本文（Markdown）
  joseki: string | null      # 検出された定跡名
  matched_position: string | null  # 有名局面名
  relevant_proverbs: string[]      # 適用された格言
  backend_used: string       # 実際に使用したバックエンド
  model_used: string         # 実際に使用したモデル名
```

---

## フロントエンド連携

既存 `web/index.html` に解説パネルを追加する。

- 「解説」ボタン押下 → `/explain` を呼び出し
- 解説テキストをサイドパネルに Markdown レンダリング
- LLMバックエンド・解説レベルをUIから切り替え可能

---

## 実装フェーズ

### Phase 1（コア：2〜3日）
- [ ] `context_builder.py`: SFEN → 盤面テキスト・候補手テキスト化
- [ ] `commentator.py`: Ollama / Claude API 切り替え対応
- [ ] `api.py`: `/explain` エンドポイント
- [ ] データなし（格言・定跡はLLMの学習知識に委ねる）

### Phase 2（知識ベース：+1〜2日）
- [ ] `data/proverbs.json`: 格言50〜100件
- [ ] `data/joseki_map.json`: 主要戦型判定（矢倉・振り飛車・穴熊・雁木等）
- [ ] `data/famous_positions.json`: 有名局面20〜30件
- [ ] `knowledge.py`: 上記DBを使った照合ロジック

### Phase 3（統合：+1日）
- [ ] フロントエンド解説パネル実装
- [ ] `solve/` モジュール連携（詰みがある局面では詰み手順も解説）
- [ ] ストリーミング解説（体験向上）

---

## 依存関係

```toml
# pyproject.toml に追加予定
ollama = ">=0.3"         # Ollama Python クライアント
anthropic = ">=0.30"     # Claude API（オプション）
markdown = ">=3.6"       # レスポンスのMarkdown処理
```

既存依存: `fastapi`, `shogi`, `core/board.py` をそのまま利用。
