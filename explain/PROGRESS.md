# PROGRESS.md — 盤面解説AI 実装進捗

## 完了済み（Phase 1）

### 作成ファイル
- `explain/__init__.py` — モジュール宣言
- `explain/context_builder.py` — SFEN+候補手 → LLMプロンプト用テキスト変換
- `explain/commentator.py` — OllamaBackend / ClaudeBackend + Commentator クラス
- `explain/api.py` — POST /explain FastAPIルーター
- `explain/README.md` — モジュール概要・構成・フェーズ計画
- `explain/DESIGN.md` — 詳細設計（データフロー・API仕様・JSON構造・懸念事項）
- `explain/data/` — 空ディレクトリ（Phase 2 でJSONを配置）

### 変更ファイル
- `pyproject.toml` — commentary extras に `httpx` 追加
- `web/app.py` — explain router 組み込み、state レスポンスに `sfen` フィールド追加
- `web/index.html` — marked.js CDN追加・解説パネルUI（CSS+HTML+JS）追加

### 動作確認済み
- `POST /explain` エンドポイントが正しくルーティングされる
- Ollama未起動時は `503 LLMエラー` を返す（正常動作）
- `GET /api/state` に `sfen` フィールドが含まれる
- context_builder が初期局面・候補手・評価値テキストを正しく生成

---

## 次のステップ（Phase 2）

### やること
1. `explain/data/proverbs.json` — 将棋格言50〜100件のJSONを作成
2. `explain/data/joseki_map.json` — 主要戦型の代表パターン（矢倉・振り飛車・穴熊・雁木等）
3. `explain/data/famous_positions.json` — 有名局面20〜30件（SFEN + 棋戦名）
4. `explain/knowledge.py` — 上記DBを使った戦型判定・格言マッチ・有名局面照合ロジック
5. `explain/api.py` — レスポンスに `joseki`, `matched_position`, `relevant_proverbs` フィールド追加
6. `explain/commentator.py` — knowledge情報をプロンプトに組み込む

### Phase 3 候補（優先度低）
- ストリーミング解説（FastAPI StreamingResponse）
- `solve/` モジュールと連携して詰み局面では詰み手順も解説に含める
- フロントエンド解説パネルに定跡名・格言を別枠で表示
