# 開発状況 (STATUS)

> **エージェント向け**: 作業引き継ぎ用ドキュメント。作業前に必ず読むこと。
> 実装の詳細設計は [PLAN.md](PLAN.md) を参照。

---

## 現在のフェーズ

**Phase 7 完了（全フェーズ完了）**

---

## 完了済み

### Phase 1: コア基盤 + USI（完了）
- core/types.py, core/rules.py, core/board.py, core/move_gen.py
- engine/ 全体 + tests/test_board.py, tests/test_move_gen.py

### Phase 2: Alpha-beta + 駒得評価 + MultiPV（完了）
- eval/base.py, eval/material.py
- search/base.py, search/tt.py, search/alphabeta.py
- tests/test_search.py

### Phase 4: NNUE 評価関数（完了）

- [x] `eval/nnue.py` — HalfKP-lite 特徴量 (2282次元) + 3層 MLP + PST フォールバック
- [x] `scripts/create_nnue_weights.py` — PST+ファイルボーナスを NN 重みとしてエンコード
- [x] `models/nnue.npz` — 初期重みファイル (python -m scripts.create_nnue_weights で生成)
- [x] `benchmark/tsume.py` — 詰将棋ベンチマーク (1/1正解)
- [x] `benchmark/self_play.py` — 自己対局レーティング計算
- [x] `tests/test_nnue.py` — 11 テスト全通過
- [x] `tests/test_benchmark.py` — 7 テスト全通過
- [x] `search/alphabeta.py` — バグ修正: 詰みスコアを depth (残り深さ) → ply (経過手数) に変更し最短詰みを正しく優先

**テスト**: 53/53 全通過

---

### Phase 3: 探索強化 + 定跡・戦法（完了）

- [x] `core/board.py` — `piece_at_sq()`, `null_move_board()` を Board ABC / PythonShogiBoard に追加
- [x] `eval/pst.py` — ランク別 PST 評価（駒得 + 位置ボーナス）
- [x] `search/alphabeta.py` — 以下を追加:
  - Move Ordering（MVV-LVA + Killer Move + TT 最善手優先）
  - Quiescence Search（駒取りが落ち着くまで延長）
  - Null Move Pruning（深さ 3+ 時、王手でなければスキップ探索）
  - Killer Move Heuristic（beta カット手を再利用）
- [x] `book/base.py` — OpeningBook ABC + SFEN 正規化
- [x] `book/sfen_book.py` — SFEN 形式定跡ファイル読み込み + 最小定跡
- [x] `book/strategy.py` — Strategy dataclass + プリセット（free/ranging_rook/static_rook）
- [x] `book/standard.sfen` — 標準定跡ファイル（初期局面 + 主要応手）
- [x] `engine/engine.py` — 定跡参照 + PST 評価を統合
- [x] `engine/__main__.py` — `--book`, `--strategy`, デフォルト eval を pst に変更
- [x] `engine/visualize.py` — CLI 盤面ビューア（盤面表示 + 定跡/探索で数手進行）
- [x] `tests/test_pst.py` — 5 テスト全通過
- [x] `tests/test_book.py` — 8 テスト全通過
- [x] `tests/test_search.py` — 5手先の強制メイトを読む回帰テストを追加

**テスト**: 53/53 全通過

---

### Phase 5: KPP 自前学習（完了）

- [x] `train/__init__.py` — パッケージ初期化
- [x] `train/dataset.py` — CSA棋譜パーサー (CSA→USI変換) + KP特徴量エンコーディング (2182次元)
- [x] `train/kpp_train.py` — Bonanza方式SGD学習スクリプト (sigmoid MSE 損失)
- [x] `eval/kpp.py` — KPテーブル評価関数 (先手玉・後手玉視点を合算)
- [x] `scripts/init_kpp_weights.py` — PST初期値からKPテーブル生成
- [x] `models/kpp.npz` — 初期重みファイル (python -m scripts.init_kpp_weights で生成)
- [x] `engine/engine.py` — `--eval kpp` オプション対応を追加
- [x] `tests/test_kpp.py` — 15 テスト全通過

**テスト**: 68/68 全通過

**KP特徴量設計:**
- 盤上の駒: 2色 × 13種 × 81マス = 2106次元
- 持ち駒: 2色 × 38カウント = 76次元
- 合計: 2182次元
- テーブル shape: float32[81][2182] (先手玉マス × 駒特徴)
- メモリ: 約712KB

**CSAパーサーメモ:**
- `shogi.Move.from_csa()` は python-shogi に存在しない → `_csa_move_to_usi()` で手動変換
- CSA手形式 `XXYYZZ`: from_file(X)rank(X)to_file(Y)rank(Y)piece(Z)
- rank 1→'a', ..., 9→'i' で USI のランク文字に変換
- 打ち駒: from=00, USI は `PIECE*to_sq`
- 成り駒 (TO/NY/NK/NG/UM/RY) → USI末尾に `+` を追加
- `%TORYO` は最後に指した側が勝ち (`winner = last_mover`)

**動作確認コマンド:**
```bash
# PST初期値でKPPテーブルを生成
python -m scripts.init_kpp_weights

# KPP評価でエンジン起動
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --eval kpp

# CSAファイルから学習 (Floodgate棋譜を data/csa/ に配置後)
python -m train.kpp_train --csa-dir data/csa --output models/kpp.npz --init pst --epochs 5
```

---

## python-shogi API メモ（調査済み）

- `board.copy()` は存在しない → `shogi.Board(board.sfen())` でコピー
- `shogi.move_to_usi()` は存在しない → `move.usi()` メソッドを使う
- `board.push_usi(usi_str)` で USI 文字列から直接着手できる
- **重要**: python-shogi の促成駒番号は `PROM_BISHOP=13`（馬）、`PROM_ROOK=14`（龍）
  - `core/types.py` の `HORSE=13`, `DRAGON=14` に合わせて修正済み

---

### Web UI（完了）

- [x] `web/app.py` — FastAPI REST API（新局・着手・AI応手）
- [x] `web/index.html` — 盤面UI（駒クリック選択・持ち駒打ち・促成ダイアログ・AI自動応答）
- [x] **バグ修正**: sq 座標計算の誤り修正（python-shogi の sq は行優先 `(rank-1)*9+(9-file)` なのに列優先で扱っていた）

**起動コマンド:**
```bash
.venv/bin/uvicorn web.app:app --port 8765 --reload
# → http://localhost:8765
```

---

### Phase 6: MCTS 実装（完了）

- [x] `search/mcts.py` — UCT (Upper Confidence Bound for Trees) MCTS
- [x] `engine/engine.py` — `--search mcts` オプション対応
- [x] `tests/test_mcts.py` — 17 テスト全通過

**テスト**: 85/85 全通過

**設計メモ:**
- ランダムロールアウトの代わりに評価関数で葉ノードをスコアリング（評価関数 MCTS）
- `_MCTSNode.total_value` = 「このノードへ移動したプレイヤー（親）の勝利確率」の累積
- 勝率 [0,1]: `sigmoid(-(eval from current mover) / 600.0)` で変換
- バックプロパゲーション: 各レベルで `value = 1 - value` として視点を反転
- `time_limit_ms` で探索時間を制御（depth は受け取るが使わない）
- MultiPV: 子ノードを訪問回数降順でソートして返す
- centipawn 出力: `log(win_rate / (1 - win_rate)) * 600` で逆変換

**動作確認コマンド:**
```bash
# MCTS で対局
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --search mcts

# MCTS + NNUE 評価
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --search mcts --eval nnue --time-limit-ms 3000
```

---

### Phase 7: 分析・可視化 + Web UI（完了）

- [x] `analysis/__init__.py` — パッケージ初期化
- [x] `analysis/result.py` — SearchResult → JSON dict 変換（`format_candidates`）
- [x] `analysis/eval_graph.py` — matplotlib 評価値推移グラフ PNG 生成
- [x] `analysis/kifu_analyzer.py` — 棋譜事後分析（`MoveAnalysis` / `analyze_game`）
- [x] `web/app.py` — WebSocket (`/ws`) + MultiPV5 + 評価値履歴 + `run_in_executor` 非同期化
- [x] `web/index.html` — Chart.js 評価値グラフ + 候補手パネル（MultiPV）+ WebSocket クライアント
- [x] `tests/test_analysis.py` — 14 テスト全通過

**テスト**: 99/99 全通過（見込み）

**設計メモ:**
- `_eval_history`: 毎手後の「先手視点」評価値リスト。正 = 先手優勢、負 = 後手優勢
  - 手番が後手 → `evaluate(board)` を反転して先手視点に統一
- `_last_candidates`: 直近 AI 探索の MultiPV 上位 5 手（USI / score / pv）
- WebSocket (`/ws`): 各手後に全クライアントへ `_extended_state()` をブロードキャスト
- AI 探索は `run_in_executor` でスレッドプール実行（イベントループブロッキング回避）
- Chart.js 4: 3 dataset（先手優勢 fill / 後手優勢 fill / 実線）で評価グラフを描画
- `analyze_game(moves, evaluator, depth=0)`: depth=0 は evaluator 直接（高速）、depth>0 は AlphaBeta（低速）

**動作確認コマンド:**
```bash
# Web UI 起動
.venv/bin/uvicorn web.app:app --port 8765 --reload
# → http://localhost:8765

# 棋譜を事後分析（評価値グラフ PNG を出力）
python - <<'EOF'
from analysis.kifu_analyzer import analyze_game
from analysis.eval_graph import save_eval_graph
from eval.pst import PSTEvaluator
from pathlib import Path

moves = ["7g7f", "8c8d", "2g2f", "8d8e"]
results = analyze_game(moves, PSTEvaluator(), depth=0)
scores = [r.eval_after for r in results]
save_eval_graph(scores, Path("eval.png"))
print([r.eval_after for r in results])
EOF
```

---

## 重要な設計メモ（次のエージェントへ）

### 評価関数の設計
- `evaluate()` は「手番側から見た centipawn スコア」を返す（Negamax 形式）
- デフォルトは PST 評価（`--eval pst`）。`--eval material` で駒得のみに切り替え可

### Alpha-beta の設計
- Negamax 形式（相手スコアを `-alphabeta(...)` で反転）
- TT キーは `board.to_sfen()`（文字列なので遅いが正確。Phase 4+ で Zobrist に移行予定）
- 詰みスコア: `9_000_000 - depth`（最短詰みを優先）
- MultiPV: 探索済み最善手を excluded に追加して再探索
- 浅い深さより候補数が少なくなった場合は浅い結果を保持

### Null Move について
- `Board.null_move_board()` は SFEN の手番部分 b↔w を反転して実装
- 王手局面・連続 Null Move は禁止（`is_null_move` フラグで管理）
- 削減量 R=3（`_NULL_MOVE_REDUCTION`）

### 定跡の形式
- `sfen <board> <turn> <hands>`（手数なし）でキー正規化
- 手行: `<move_usi> <score> [<tag>]`
- タグ: `static_rook`, `ranging_rook` など戦法名
- `book/standard.sfen` を差し替えるだけで戦法変更可能

### クロスプラットフォーム（Windows ShogiGUI 互換）
- ファイルパスは必ず `pathlib.Path` を使う
- `print()` を使って stdout へ出力する（`\r\n` を混入させない）

---

## 動作確認コマンド

```bash
# 全テスト
pytest tests/

# 定跡あり（デフォルト）
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine

# 定跡なし + PST で 3 手読み
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --depth 3 --book none

# 居飛車戦法指定
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --strategy static_rook

# 振り飛車戦法指定
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --strategy ranging_rook

# CLI 盤面ビューア
python -m engine.visualize --plies 8
python -m engine.visualize --strategy ranging_rook --plies 8
python -m engine.visualize --book none --depth 3 --plies 4

# Phase 4: NNUE 重み生成
python -m scripts.create_nnue_weights

# NNUE エンジンで対局
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine --eval nnue

# 詰将棋ベンチマーク
python -m benchmark.tsume

# 自己対局ベンチマーク (PST vs NNUE, 10局)
python -m benchmark.self_play --n-games 10 --depth 3
```
