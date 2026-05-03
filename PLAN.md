# 将棋AI 開発計画

## 概要

Python で実装する将棋AIエンジン。複数の探索アルゴリズム・評価関数を設定で切り替え可能な設計とし、アマ三〜五段相当の強さを目標とする。

- **言語**: Python 3.12+（型ヒント strict、mypy 適用）
- **目標強度**: アマ三〜五段（KPP → NNUE の順に実装・比較）
- **用途**: 技術研究・将棋上達支援・将来的な Hi!story AI への技術転用
- **運用方針**: AIエージェントが主体的に実装・修正できる構造にする

---

## 設計の大原則

**「まず動かす、後から差し替える」**

- 最初は `python-shogi` を使って最速で動かす
- 各コンポーネントは ABC（抽象基底クラス）で定義し、実装を後から差し替えられる
- 将棋以外のルール・盤面への拡張を最初から想定した設計にする

### 抽象化の対象

| コンポーネント | 初期実装 | 後から差し替えられるもの |
|--------------|---------|------------------------|
| 盤面・手生成 | `python-shogi` ラッパー | 自作合法手生成エンジン |
| 評価関数 | 駒得 → KPP → NNUE | 独自評価、Hi!story 用評価 |
| 探索アルゴリズム | Alpha-beta | MCTS、その他 |
| ルールセット | 標準将棋 | 変則ルール、Hi!story ルール |
| 解説生成 | なし | LLM（Claude API など） |

---

## 強さへの経路

```
Phase 1: python-shogi ラッパー + USI    → 合法手を指せる（最短で動く）
Phase 2: Alpha-beta + 駒得評価          → ランダムAIには必ず勝てる
Phase 3: 探索強化（手順・置換表）        → 5級〜初段相当
Phase 4: 既存 NNUE 重みを流用           → 三段〜五段相当  ← 最初の目標
Phase 5: KPP 自前学習                   → NNUE との比較実験
Phase 6: MCTS 実装                      → 探索手法の比較・Hi!story 布石
Phase 7: 分析・可視化ツール + Web UI    → ブラウザで対局・グラフ表示
```

### 評価関数の比較（後から差し替え可能）

| 手法 | 強さの目安 | 特徴 |
|------|-----------|------|
| 駒得のみ | 3〜5級 | 実装が最もシンプル |
| PST（位置ボーナス） | 1〜2級 | 手作業チューニング |
| **KPP**（Bonanza 方式） | **初段〜三段** | 棋譜から自動学習 |
| **NNUE**（既存重み流用） | **三段〜五段** | 即日で強い。後から自前学習も可 |

---

## ディレクトリ構成

```
shogi-ai/
├── PLAN.md
├── pyproject.toml
│
├── core/
│   ├── types.py          # 駒・色・マス・手の型定義（全モジュールが依存）
│   ├── board.py          # Board ABC + PythonShogiBoard（python-shogi ラッパー）
│   ├── move_gen.py       # MoveGenerator ABC + PythonShogiMoveGen
│   ├── rules.py          # RuleSet dataclass（標準・変則ルール切り替え）
│   └── zobrist.py        # Zobrist ハッシュ（置換表用）
│
├── search/
│   ├── base.py           # Searcher ABC + SearchResult + CandidateMove 型
│   ├── minimax.py        # Minimax（テスト・比較用）
│   ├── alphabeta.py      # Alpha-beta + 反復深化 + MultiPV
│   ├── mcts.py           # Monte Carlo Tree Search（Phase 6）
│   └── tt.py             # 置換表（Zobrist ハッシュベース）
│
├── eval/
│   ├── base.py           # Evaluator ABC
│   ├── material.py       # 駒得のみ
│   ├── pst.py            # 駒得 + 位置ボーナステーブル
│   ├── kpp.py            # KPP 評価関数（Phase 5）
│   └── nnue.py           # NNUE（ONNX Runtime 推論）（Phase 4）
│
├── engine/
│   ├── engine.py         # Searcher + Evaluator + RuleSet を組み合わせる本体
│   ├── config.py         # EngineConfig dataclass（手法・ルール切り替え）
│   └── usi.py            # USI プロトコル（MultiPV 対応）
│
├── train/
│   ├── kpp_train.py      # KPP テーブルの学習（Bonanza 方式）
│   ├── nnue_model.py     # PyTorch NNUE モデル定義（MPS 対応）
│   ├── dataset.py        # KIF/CSA 棋譜読み込み・特徴量変換
│   └── nnue_train.py     # NNUE 学習スクリプト
│
├── benchmark/
│   ├── tsume.py          # 詰将棋ベンチマーク（強さ測定）
│   └── self_play.py      # 自己対局レーティング計算
│
├── analysis/
│   ├── result.py         # SearchResult → 候補手・評価値の整形
│   ├── eval_graph.py     # 評価値推移グラフ生成（matplotlib）
│   └── kifu_analyzer.py  # KIF ファイルの事後分析
│
├── book/
│   ├── base.py           # OpeningBook ABC
│   ├── sfen_book.py      # SFEN 形式の定跡ファイル読み込み（既存ファイル流用可）
│   └── strategy.py       # Strategy dataclass（戦法の定義・制約）
│
├── commentary/
│   ├── base.py           # Commentator ABC
│   └── llm.py            # LLM 解説（Claude API）（将来実装）
│
└── web/
    ├── app.py            # FastAPI アプリ（WebSocket + REST）
    ├── game_session.py   # 対局セッション管理
    └── static/
        ├── index.html    # 対局画面
        ├── board.js      # SVG 盤面描画・駒操作
        └── graph.js      # 評価値グラフ（Chart.js）
```

---

## アーキテクチャ設計

### 手法切り替え（Strategy パターン）

```python
# engine/config.py
@dataclass
class EngineConfig:
    search:       Literal["minimax", "alphabeta", "mcts"] = "alphabeta"
    eval:         Literal["material", "pst", "kpp", "nnue"] = "nnue"
    board_impl:   Literal["python_shogi", "native"] = "python_shogi"
    depth:        int = 5
    time_limit_ms: int = 3000
    multi_pv:     int = 5
    nnue_model_path:   str = "models/nnue.onnx"
    kpp_table_path:    str = "models/kpp.bin"
    opening_book_path: str | None = "book/standard.sfen"  # None で定跡なし
    strategy:          str | None = None  # "ranging_rook" / "static_rook" など
    rules:        RuleSet = field(default_factory=RuleSet)
```

### ルールセット（拡張性の核心）

```python
# core/rules.py
@dataclass
class RuleSet:
    # 標準ルールからの変更フラグ
    allow_double_pawn:              bool = False   # 二歩あり
    allow_pawn_on_last_rank:        bool = False   # 端歩なし（成れない位置の歩を打てる）
    king_moves_only_when_in_check:  bool = False   # 取れる時しか玉を動かせない
    allow_arbitrary_start:          bool = False   # 任意の初期配置（変な盤面）
    # 将来: Hi!story 用フラグなど
```

`MoveGenerator` は `RuleSet` を受け取って合法手を生成する。ルール変更は `RuleSet` のフラグを変えるだけ。

### 盤面の抽象化（python-shogi の差し替え可能化）

```python
# core/board.py
class Board(ABC):
    @abstractmethod
    def apply_move(self, move: Move) -> "Board": ...
    @abstractmethod
    def is_check(self) -> bool: ...
    @classmethod
    @abstractmethod
    def from_sfen(cls, sfen: str) -> "Board": ...  # 変な盤面も SFEN で渡せる

class PythonShogiBoard(Board):
    """python-shogi ラッパー。後から NativeBoard に差し替え可能"""
    ...
```

### MultiPV と評価値の流れ

```
Searcher.search() → SearchResult
    ├── best_move: Move
    ├── best_score: int
    ├── candidates: list[CandidateMove]  # 上位 N 手
    └── depth, nodes

engine/usi.py → info multipv 1..N score cp XXX pv ... を標準出力へ
web/game_session.py → WebSocket でブラウザへ push
analysis/eval_graph.py → matplotlib グラフ生成
```

### モジュール依存の方向

```
core（types, board, move_gen, rules, zobrist）
    ↓
eval（base, material, pst, kpp, nnue）
search（base, minimax, alphabeta, mcts, tt）
    ↓
engine（engine, config, usi）
    ↓
benchmark（tsume, self_play）
analysis（result, eval_graph, kifu_analyzer）
commentary（base, llm）
web（app, game_session）

train（model, dataset, train） ← core + eval のみ依存（独立）
```

循環依存なし。core が最上流。

---

## フェーズ別実装計画

### Phase 1: python-shogi ラッパー + USI（3〜5日）

**目標**: 最速で合法手を指せる状態にする

| ファイル | 内容 |
|---------|------|
| `core/types.py` | `Piece`, `Color`, `Square`, `Move` 型定義 |
| `core/rules.py` | `RuleSet` dataclass（標準ルールのみ）|
| `core/board.py` | `Board` ABC + `PythonShogiBoard`（python-shogi ラッパー） |
| `core/move_gen.py` | `MoveGenerator` ABC + `PythonShogiMoveGen` |
| `engine/config.py` | `EngineConfig` dataclass |
| `engine/usi.py` | USI ループ（usi / isready / position / go / stop） |
| `tests/test_board.py` | 初期配置・SFEN 読み込みの検証 |
| `tests/test_move_gen.py` | 初期局面の手数・特定局面での合法手を検証 |

**完了基準**: `pytest tests/` が全通過 + USI ループが CLI で正常動作

---

### Phase 2: Alpha-beta + 駒得評価（1週間）

**目標**: ランダムAIに必ず勝てる + MultiPV の骨格を作る

| ファイル | 内容 |
|---------|------|
| `eval/base.py` | `Evaluator` ABC |
| `eval/material.py` | 駒の点数テーブルによる評価 |
| `search/base.py` | `Searcher` ABC + `SearchResult` + `CandidateMove` 型 |
| `search/tt.py` | 置換表（TranspositionTable） |
| `search/alphabeta.py` | Alpha-beta + 反復深化 + MultiPV（上位 N 手） |
| `engine/engine.py` | Searcher + Evaluator を組み合わせる |
| `engine/usi.py` | MultiPV オプション対応・`info multipv` 送信追加 |
| `tests/test_search.py` | 1手詰め・3手詰めの正解手チェック |

**完了基準**: 3手詰めを解ける + ランダムAIに 10 戦全勝

---

### Phase 3: 探索強化 + 定跡・戦法（1〜2週間）

**目標**: 5級〜初段相当 + 定跡・戦法を指定した対局ができる

#### 探索強化

| 技術 | 内容 |
|------|------|
| Move Ordering | 取る手・王手を優先探索 |
| Quiescence Search | 駒の取り合いが収束するまで延長 |
| Null Move Pruning | 探索木の枝刈り強化 |
| Killer Move Heuristic | β カット手を優先再利用 |
| `eval/pst.py` | 位置ボーナステーブルで評価精度向上 |

#### 定跡・戦法

| ファイル | 内容 |
|---------|------|
| `book/base.py` | `OpeningBook` ABC（定跡参照インターフェース） |
| `book/sfen_book.py` | SFEN 形式の定跡ファイル読み込み（YaneuraOu の定跡ファイルを流用可） |
| `book/strategy.py` | `Strategy` dataclass + 戦法フィルター（振り飛車・居飛車など） |
| `engine/engine.py` | 序盤は定跡を参照、定跡を外れたら探索に切り替え |

```python
# book/strategy.py
@dataclass
class Strategy:
    name: str                    # "ranging_rook" / "static_rook" / "free" など
    allow_openings: list[str]    # 許可する定跡分類
    # 「振り飛車しか指さない」「矢倉に誘導する」などを定義
```

定跡ファイルは YaneuraOu 形式（SFEN + スコア）をそのまま読めるよう実装し、
ファイルを差し替えるだけで戦法を変更できる。

**完了基準**: 5手詰めを安定して解ける + 「振り飛車モード」で定跡通りに序盤を指せる

---

### Phase 4: 既存 NNUE 重みを流用（1週間）

**目標**: 三段〜五段相当を最短で達成する

既存のオープンソース将棋エンジン（水匠・YaneuraOu など）の NNUE 重みを変換・流用する。自前学習は Phase 5 以降。

| ファイル | 内容 |
|---------|------|
| 重み変換スクリプト | 既存 NNUE 重みを ONNX 形式に変換 |
| `eval/nnue.py` | ONNX Runtime で推論・`Evaluator` ABC 実装 |
| `benchmark/tsume.py` | 詰将棋ベンチマーク（強さ測定） |
| `benchmark/self_play.py` | 自己対局レーティング計算 |

**完了基準**: Phase 3 エンジンに 9 割以上勝利 + 詰将棋ベンチマークで強さを確認

---

### Phase 5: KPP 自前学習（2〜3週間）

**目標**: NNUE との実力比較・学習の仕組みを理解する

| ファイル | 内容 |
|---------|------|
| `train/dataset.py` | CSA 棋譜（Floodgate）読み込み・KPP 特徴量生成 |
| `train/kpp_train.py` | KPP テーブル学習（Bonanza 方式 SGD） |
| `eval/kpp.py` | 学習済みテーブルのロード + 評価実装 |

**学習データ**: Floodgate（CSA 形式、無料・大量）

**完了基準**: KPP が Phase 3 エンジン（PST 評価）に 9 割勝利

---

### Phase 6: MCTS 実装（2週間）

**目標**: Alpha-beta との探索手法比較・Hi!story AI の基礎

| ファイル | 内容 |
|---------|------|
| `search/mcts.py` | UCT（Upper Confidence Bound for Trees） |
| ロールアウト評価 | material / NNUE を切り替えて比較 |

MCTS はルールが変わっても汎化しやすいため Hi!story AI への応用に有利。変則ルール実験にも使いやすい。

**完了基準**: Alpha-beta と同じ時間制限で対局して比較実験が回せる

---

### Phase 7: 分析・可視化 + Web UI（2〜3週間）

**目標**: ブラウザで対局でき、候補手 5 本と評価値グラフがリアルタイム表示される

#### 機能一覧

| 機能 | 実装箇所 | 出力先 |
|------|---------|--------|
| 候補手 N 本と評価値 | `search/alphabeta.py` MultiPV | Web UI リアルタイム表示 |
| 評価値推移グラフ | `web/static/graph.js`（Chart.js） | ブラウザ内リアルタイム更新 |
| 棋譜事後分析 | `analysis/kifu_analyzer.py` | matplotlib PNG + テキスト出力 |
| SVG 盤面・駒操作 | `web/static/board.js` | ドラッグ&ドロップで操作 |
| WebSocket 通信 | `web/app.py`（FastAPI） | ブラウザ ↔ エンジン双方向 |

#### Web UI のアーキテクチャ

```
ブラウザ（board.js + graph.js）
    ↕ WebSocket
web/app.py（FastAPI）
    ↕ 関数呼び出し
engine/engine.py（Searcher + Evaluator + RuleSet）
    ↓
analysis/result.py → グラフデータを WebSocket でブラウザへ push
```

**完了基準**: `http://localhost:8000` でブラウザから AI と対局でき、候補手 5 本と評価値グラフがリアルタイム表示される

---

## 将来の拡張計画

### 定跡・戦法の拡張

定跡ファイルは差し替えるだけで戦法を変更できる。ユーザーが自分で定跡ファイルを作成・編集して研究にも使える。

```bash
# 振り飛車で対局
python -m engine --strategy ranging_rook --book book/ranging_rook.sfen

# 定跡なし（純粋な探索のみ）
python -m engine --book none

# 自分の研究定跡を登録して試す
python -m engine --book book/my_research.sfen
```

戦法フィルターは「この戦法の局面にしか定跡を適用しない」という制約として機能し、
定跡を外れた後は通常の探索に自然に切り替わる。

---

### 変則ルール対応（RuleSet 拡張）

`RuleSet` にフラグを追加するだけで対応できる設計。`MoveGenerator` の実装を変えなくてよい。

```python
# 例: 変則ルールの設定
config = EngineConfig(
    rules=RuleSet(
        allow_double_pawn=True,              # 二歩あり
        allow_pawn_on_last_rank=False,       # 端歩なし
        king_moves_only_when_in_check=True,  # 取れる時しか玉を動かせない
        allow_arbitrary_start=True,          # 全部持ち駒の局面も可
    )
)
```

### 変な盤面への対応

`Board.from_sfen(sfen)` がどんな SFEN 文字列も受け付ける設計にする。
評価関数は「駒の点数が計算できる」ことだけを前提とするため、標準外の盤面でも動く。

### LLM 解説機能

`commentary/` モジュールに `Commentator` ABC を置き、実装を後から追加する。

```python
# commentary/base.py
class Commentator(ABC):
    @abstractmethod
    def comment(self, board: Board, result: SearchResult) -> str: ...

# commentary/llm.py（将来実装）
class ClaudeCommentator(Commentator):
    """Claude API を使って局面の解説文を生成"""
    ...
```

### Hi!story AI への応用

| 将棋AIの技術 | Hi!story AIへの転用 |
|-------------|-------------------|
| `Board` ABC + `RuleSet` | Hi!story 独自ルールの盤面・アクション管理 |
| Alpha-beta | カードゲームの先読み |
| **MCTS** | **Hi!story ルールへの汎化（最も有力）** |
| KPP / NNUE | 対戦データからの自動評価学習 |
| Web UI | ブラウザ上でのデジタル対戦 |
| LLM 解説 | カード効果・戦略の解説 |

---

## 技術選定

| 項目 | 選択 | 理由 |
|------|------|------|
| 盤面・手生成（初期） | python-shogi ラッパー | 最速で動かす。ABC の後ろに隠すので差し替え可能 |
| 盤面・手生成（後期） | 自作（NativeBoard） | Phase 5 以降で差し替え。技術理解のため |
| NNUE 推論 | ONNX Runtime | PyTorch より推論速度が速い。依存が軽い |
| NNUE 学習 | PyTorch（MPS バックエンド） | M5 Mac の GPU 加速を活用 |
| KPP 学習 | NumPy + 手実装 SGD | 外部依存なし |
| 棋譜データ | Floodgate（CSA 形式） | 無料・大量・高段者棋譜 |
| 可視化（CLI） | matplotlib | 棋譜事後分析・PNG 保存 |
| Web フレームワーク | FastAPI | WebSocket 対応・型安全・軽量 |
| フロントエンド | SVG + Chart.js（CDN） | フレームワークなし。依存最小 |
| テスト | pytest | シンプル |
| 設定 | dataclass + TOML | 型安全 |

---

## 依存パッケージ

```toml
# pyproject.toml
[project]
name = "shogi-ai"
requires-python = ">=3.12"

dependencies = [
  "python-shogi",   # 合法手生成・KIF/CSA パーサー（Phase 1〜。後から差し替え可）
  "numpy",          # 盤面演算・KPP 特徴量計算
  "onnxruntime",    # NNUE 推論（Phase 4〜）
  "matplotlib",     # 棋譜事後分析グラフ（Phase 7〜）
  "fastapi",        # Web UI サーバー（Phase 7〜）
  "uvicorn",        # FastAPI ASGI サーバー
]

[project.optional-dependencies]
train = [
  "torch",          # NNUE / KPP 学習（train/ のみ使用。M5 は MPS バックエンド）
]
commentary = [
  "anthropic",      # LLM 解説機能（将来実装）
]
dev = [
  "pytest",
  "mypy",
]
```

---

## AIエージェント実装ガイドライン

### クロスプラットフォーム対応（Windows ShogiGUI との互換性）

USI プロトコルは stdin/stdout のテキスト通信なので、Python が動く Windows 環境であれば ShogiGUI からそのまま呼び出せる。ただし以下 2 点を守ること。

| 規則 | 内容 |
|------|------|
| パス区切り | ファイルパスは `pathlib.Path` を使う。`/` ハードコード禁止 |
| 改行コード | `sys.stdout` への出力は `print()` を使う（`\r\n` を混入させない）。USI 仕様は LF |

Web UI・matplotlib など GUI 系ツールは Mac 専用でよい。`engine/usi.py` と `core/` 配下はプラットフォーム非依存を保つ。

---

### コーディング規約

| 規約 | 内容 |
|------|------|
| 型ヒント | 全関数・メソッドに必須（mypy strict モード適合） |
| 依存方向 | `core` → `eval`/`search` → `engine` → 下流モジュール を厳守 |
| グローバル状態 | 持たない。`Board.apply_move` は新インスタンスを返す |
| マジックナンバー | `core/types.py` の定数に集約 |
| コメント | WHY が自明でない箇所のみ |
| エラー処理 | 境界値チェックはテストで担保。防衛的 assert 不要 |
| 差し替え前提 | `python-shogi` への直接依存は `core/board.py` と `core/move_gen.py` のみ |

### テスト方針

- **合法手生成**: 初期局面の手数・王手局面・打ち歩詰め・変則ルール
- **探索**: 1手詰め・3手詰め・5手詰めの正解手チェック
- **MultiPV**: 返される候補手数が指定通りか・スコアが降順か
- **評価**: 明らかに優勢な局面のスコアが正の値であること
- Phase ごとに `pytest tests/` が全通過することを完了条件とする

### 実装の進め方

1. ABC のインターフェースを先に定義し、次に実装クラスを書く
2. 1モジュール = 1責務。200 行を超えたら分割を検討
3. 新しい手法は継承・差し込みで追加し、既存クラスを変更しない
4. `python-shogi` 依存は `core/board.py` と `core/move_gen.py` だけに閉じ込める
