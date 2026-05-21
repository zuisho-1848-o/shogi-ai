# shogi-ai

Python 製の将棋AIエンジン。複数の探索アルゴリズム・評価関数を切り替えて研究に使える。

**テスト**: 99/99 全通過 | **フェーズ**: 1〜7 全完了

## 特徴

- **複数の手法を切り替え可能**: Alpha-beta / MCTS + 駒得 / PST / KPP / NNUE
- **USI 対応**: CLI で動作、ShogiGUI（Windows）からも使える
- **Web UI**: ブラウザで人間 vs AI 対局・リアルタイム評価値グラフ・MultiPV候補手表示
- **棋譜分析**: 各手の評価値・候補手を事後分析、PNG グラフ出力
- **KPP 自前学習**: CSA 棋譜から評価関数を強化

---

## 必要環境

- Python 3.12+

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,dev]"
```

学習も行う場合（M5 Mac では MPS が自動使用される）:

```bash
pip install -e ".[full,train,dev]"
```

以降の操作はすべて venv を有効化した状態で行うこと。

---

## ユーザーができること

### 1. Web UIで人間 vs AI 対局（最も手軽）

```bash
.venv/bin/uvicorn web.app:app --port 8765 --reload
# → http://localhost:8765 をブラウザで開く
```

- 駒をクリックして選択 → 移動先をクリックで着手
- 持ち駒クリック → 打ち込み先を選択、成り駒はダイアログで選択
- 「AI応手」ボタンでAIが指す
- リアルタイムで**評価値グラフ**（青=先手優勢・赤=後手優勢）が更新
- **MultiPV候補手パネル**（AIが考えた上位5手をスコアつきで表示）

---

### 2. USIプロトコルでエンジン起動（ShogiGUI等との接続）

```bash
# デフォルト（PST評価・定跡あり）
python -m engine

# 評価関数・探索手法・戦法を変えて起動
python -m engine --depth 5 --eval nnue
python -m engine --eval kpp
python -m engine --search mcts --time-limit-ms 3000
python -m engine --strategy ranging_rook   # 振り飛車定跡
python -m engine --strategy static_rook    # 居飛車定跡
python -m engine --book none               # 定跡なし

# USI コマンドを手動で試す
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine
```

ShogiGUI（Windows）からの接続:
1. エンジンパスに `python` を指定
2. 引数に `-m engine` を指定（プロジェクトのルートディレクトリで実行）

---

### 3. CLI盤面ビューアで動作確認

GUI なしで、盤面を見ながら数手ぶん AI に指させられる。

```bash
python -m engine.visualize --plies 8
python -m engine.visualize --strategy ranging_rook --plies 8
python -m engine.visualize --book none --depth 3 --plies 4
```

---

### 4. 棋譜を事後分析してグラフ出力

```python
from analysis.kifu_analyzer import analyze_game
from analysis.eval_graph import save_eval_graph
from eval.pst import PSTEvaluator
from pathlib import Path

moves = ["7g7f", "8c8d", "2g2f", "8d8e"]
results = analyze_game(moves, PSTEvaluator(), depth=0)
save_eval_graph([r.eval_after for r in results], Path("eval.png"))
```

---

### 5. CSA棋譜から学習（KPP評価関数の強化）

```bash
# data/csa/ に Floodgate 等の棋譜を置いてから
python -m train.kpp_train --csa-dir data/csa --output models/kpp.npz --init pst --epochs 5
```

---

### 6. AI同士の対戦・ベンチマーク測定

評価関数・探索アルゴリズムを自由に組み合わせて対戦させられる。

```bash
# 詰将棋正答率
python -m benchmark.tsume

# PST vs NNUE（デフォルト）
python -m benchmark.self_play --n-games 10 --depth 3

# 組み合わせを自由に指定
python -m benchmark.self_play --eval1 pst --eval2 kpp --n-games 50 --depth 3
python -m benchmark.self_play --eval1 nnue --eval2 pst --search1 mcts --search2 alphabeta --n-games 20
python -m benchmark.self_play --eval1 pst --eval2 material --n-games 100 --depth 2 --quiet
```

オプション:

| オプション | 選択肢 | 説明 |
|---|---|---|
| `--eval1` / `--eval2` | `pst` `material` `nnue` `kpp` | 各エンジンの評価関数 |
| `--search1` / `--search2` | `alphabeta` `mcts` | 各エンジンの探索アルゴリズム |
| `--n-games` | 整数 | 対局数（先後を交互に入れ替えて実施） |
| `--depth` | 整数 | 探索深さ（両エンジン共通） |
| `--time-limit-ms` | ミリ秒 | 1手あたりの時間制限 |
| `--quiet` | フラグ | 局ごとの進捗表示を省略 |

> **処理負荷について**: CPU 集中処理のため、多局数・深い探索ではファンが高速回転するのは正常。目安:
> - `--depth 2` + `--n-games 10`: 数分、ファンはやや回転
> - `--depth 3` + `--n-games 50`: 10〜30分、ファンが高速回転し続ける
> - `--depth 5` 以上: 1局数時間になる場合あり
>
> 多局数の実験は `--depth 2` か `--depth 3` で行うことを推奨。`--quiet` を付けると途中出力が減る。

---

### 7. テスト実行

```bash
pytest tests/   # 99/99 全通過
```

---

## 評価関数 × 探索の組み合わせ

| 評価 | 探索 | 強さ目安 | 速度 |
|---|---|---|---|
| `material` | `alphabeta` | 入門 | 最速 |
| `pst` | `alphabeta` | 初級（デフォルト） | 速い |
| `nnue` | `alphabeta` | 中級 | 普通 |
| `kpp` | `alphabeta` | 中級〜（学習次第） | 普通 |
| `pst` | `mcts` | 初級〜中級 | 時間制限式 |
| `nnue` | `mcts` | 中級 | 時間制限式 |

---

## モジュール構成

```
shogi-ai/
├── core/          # 盤面・手生成・型定義（将棋ルールの基盤）
├── eval/          # 評価関数（material / pst / nnue / kpp）
├── search/        # 探索アルゴリズム（Alpha-beta / MCTS）
├── engine/        # エンジン本体・USI プロトコル・CLI
├── book/          # 定跡・戦法プリセット
├── train/         # KPP 学習（CSA棋譜パーサー + SGD）
├── analysis/      # 棋譜分析・評価グラフ生成
├── benchmark/     # 詰将棋ベンチマーク・自己対局
├── web/           # ブラウザ対局 UI（FastAPI + WebSocket）
├── scripts/       # 重みファイル初期生成スクリプト
├── models/        # 学習済み重みファイル（nnue.npz / kpp.npz）
└── tests/         # テスト（99ケース）
```

### 各ファイルの役割

#### `core/`
| ファイル | 役割 |
|---|---|
| [core/types.py](core/types.py) | 駒の種類・色・マス・指し手の型定義 |
| [core/board.py](core/board.py) | 盤面の抽象クラス + python-shogi ラッパー（コピー・着手・SFEN変換） |
| [core/move_gen.py](core/move_gen.py) | 合法手生成（python-shogi 委譲） |
| [core/rules.py](core/rules.py) | 王手判定・ゲーム終了判定 |

#### `eval/`
| ファイル | 評価方式 | 特徴 |
|---|---|---|
| [eval/material.py](eval/material.py) | 駒得のみ | 最速・最シンプル |
| [eval/pst.py](eval/pst.py) | 駒得 + 位置ボーナス | デフォルト。マス別に価値が変わる |
| [eval/nnue.py](eval/nnue.py) | HalfKP特徴量 + 3層MLP | ニューラルネット（2282次元） |
| [eval/kpp.py](eval/kpp.py) | KPテーブル（Bonanza方式） | CSA棋譜学習に対応 |

全評価関数は「手番側から見た centipawn スコア」を返す（Negamax形式）。

#### `search/`
| ファイル | 役割 |
|---|---|
| [search/base.py](search/base.py) | `Searcher` ABC・`SearchResult`・`CandidateMove` 型定義 |
| [search/tt.py](search/tt.py) | 置換表（Transposition Table） |
| [search/alphabeta.py](search/alphabeta.py) | Alpha-beta（MVV-LVA / Killer / Quiescence / Null Move / MultiPV） |
| [search/mcts.py](search/mcts.py) | UCT-MCTS（評価関数で葉ノードをスコアリング・時間制限制御） |

#### `engine/`
| ファイル | 役割 |
|---|---|
| [engine/engine.py](engine/engine.py) | エンジンコア（評価関数・探索・定跡を組み合わせ） |
| [engine/usi.py](engine/usi.py) | USIプロトコル I/O 処理 |
| [engine/config.py](engine/config.py) | `EngineConfig` 設定クラス |
| [engine/__main__.py](engine/__main__.py) | CLI エントリーポイント（コマンドライン引数処理） |
| [engine/visualize.py](engine/visualize.py) | CLI 盤面ビューア |

#### `analysis/`
| ファイル | 役割 |
|---|---|
| [analysis/result.py](analysis/result.py) | `SearchResult` → JSON 変換（`format_candidates`） |
| [analysis/eval_graph.py](analysis/eval_graph.py) | matplotlib で評価値推移グラフを PNG 出力 |
| [analysis/kifu_analyzer.py](analysis/kifu_analyzer.py) | 棋譜を事後分析（各手の評価値・候補手を計算） |

#### `train/`
| ファイル | 役割 |
|---|---|
| [train/dataset.py](train/dataset.py) | CSA棋譜パーサー → KP特徴量（2182次元）エンコード |
| [train/kpp_train.py](train/kpp_train.py) | Bonanza方式 SGD 学習（sigmoid MSE損失） |

#### `web/`
| ファイル | 役割 |
|---|---|
| [web/app.py](web/app.py) | FastAPI サーバー（REST API + WebSocket） |
| [web/index.html](web/index.html) | ブラウザ盤面UI（Chart.js評価グラフ・MultiPV候補手パネル） |

---

詳細な設計・実装メモは [STATUS.md](STATUS.md) と [PLAN.md](PLAN.md) を参照。
