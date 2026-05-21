# 設計メモ・実装の注意点

全フェーズ（1〜7）実装完了。テスト 99/99 全通過。

---

## 評価関数

- `evaluate()` は「手番側から見た centipawn スコア」を返す（Negamax 形式）
- 先手視点に統一する場合は `if board.turn == WHITE: score = -score`

## Alpha-beta

- Negamax 形式（相手スコアを `-alphabeta(...)` で反転）
- 置換表キーは `board.to_sfen()`（文字列なので遅いが正確）
- 詰みスコア: `9_000_000 - ply`（最短詰みを優先。depth ではなく経過手数 ply で計算）
- MultiPV: 探索済み最善手を excluded に追加して再探索

## Null Move

- `Board.null_move_board()` は SFEN の手番部分 b↔w を反転して実装
- 王手局面・連続 Null Move は禁止（`is_null_move` フラグで管理）
- 削減量 R=3（`_NULL_MOVE_REDUCTION`）

## MCTS

- `_MCTSNode.total_value` = 「このノードへ移動したプレイヤー（親）の勝利確率」の累積
- 勝率変換: `sigmoid(-(eval from current mover) / 600.0)` — evaluate() は手番側視点なので符号反転が必要
- バックプロパゲーション: 各レベルで `value = 1 - value` として視点を反転
- centipawn 逆変換: `log(win_rate / (1 - win_rate)) * 600`（win_rate は 1e-6 でクランプ）

## 定跡ファイル形式

- キー: `sfen <board> <turn> <hands>`（手数なし）
- 手行: `<move_usi> <score> [<tag>]`（タグ: `static_rook`, `ranging_rook` など）
- `book/standard.sfen` を差し替えるだけで戦法変更可能

## KPP 特徴量

- 盤上の駒: 2色 × 13種 × 81マス = 2106次元
- 持ち駒: 2色 × 38カウント = 76次元 → 合計 2182次元
- テーブル shape: `float32[81][2182]`（先手玉マス × 駒特徴）、約 712KB

---

## python-shogi API の注意点

- `board.copy()` は存在しない → `shogi.Board(board.sfen())` でコピー
- `shogi.move_to_usi()` は存在しない → `move.usi()` メソッドを使う
- `board.push_usi(usi_str)` で USI 文字列から直接着手できる
- 成り駒番号: `PROM_BISHOP=13`（馬）、`PROM_ROOK=14`（龍）→ `core/types.py` の `HORSE=13`, `DRAGON=14` に対応

## CSA棋譜パーサーの注意点

- `shogi.Move.from_csa()` は python-shogi に存在しない → `_csa_move_to_usi()` で手動変換
- CSA手形式 `XXYYZZ`: from_file(X)rank(X)to_file(Y)rank(Y)piece(Z)
- rank 1→'a', ..., 9→'i' で USI のランク文字に変換
- 打ち駒: from=00、USI は `PIECE*to_sq`
- 成り駒（TO/NY/NK/NG/UM/RY）→ USI 末尾に `+` を追加
- `%TORYO` は最後に指した側が勝ち（`winner = last_mover`）

## クロスプラットフォーム（Windows ShogiGUI 互換）

- ファイルパスは必ず `pathlib.Path` を使う
- `print()` を使って stdout へ出力する（`\r\n` を混入させない）
