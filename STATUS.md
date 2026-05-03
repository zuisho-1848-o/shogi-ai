# 開発状況 (STATUS)

> **エージェント向け**: 作業引き継ぎ用ドキュメント。作業前に必ず読むこと。
> 実装の詳細設計は [PLAN.md](PLAN.md) を参照。

---

## 現在のフェーズ

**Phase 3 完了 / Phase 4 未着手**

---

## 完了済み

### Phase 1: コア基盤 + USI（完了）
- core/types.py, core/rules.py, core/board.py, core/move_gen.py
- engine/ 全体 + tests/test_board.py, tests/test_move_gen.py

### Phase 2: Alpha-beta + 駒得評価 + MultiPV（完了）
- eval/base.py, eval/material.py
- search/base.py, search/tt.py, search/alphabeta.py
- tests/test_search.py

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
- [x] `tests/test_pst.py` — 5 テスト全通過
- [x] `tests/test_book.py` — 8 テスト全通過

**テスト**: 32/32 全通過

---

## python-shogi API メモ（調査済み）

- `board.copy()` は存在しない → `shogi.Board(board.sfen())` でコピー
- `shogi.move_to_usi()` は存在しない → `move.usi()` メソッドを使う
- `board.push_usi(usi_str)` で USI 文字列から直接着手できる
- **重要**: python-shogi の促成駒番号は `PROM_BISHOP=13`（馬）、`PROM_ROOK=14`（龍）
  - `core/types.py` の `HORSE=13`, `DRAGON=14` に合わせて修正済み

---

## Phase 4 以降（未着手）

| Phase | 内容 | 主なファイル |
|-------|------|------------|
| 4 | 既存 NNUE 重みを流用 | `eval/nnue.py`, `benchmark/` |
| 5 | KPP 自前学習 | `train/kpp_train.py`, `eval/kpp.py` |
| 6 | MCTS 実装 | `search/mcts.py` |
| 7 | 分析 + Web UI | `analysis/`, `web/` |

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
```
