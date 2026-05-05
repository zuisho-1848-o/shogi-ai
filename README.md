# shogi-ai

Python 製の将棋AIエンジン。複数の探索アルゴリズム・評価関数を切り替えて研究に使える。

## 特徴

- **複数の手法を切り替え可能**: Alpha-beta / MCTS + 駒得 / KPP / NNUE
- **USI 対応**: Mac では CLI で動作、Windows では ShogiGUI からも使える
- **研究ツール**: 候補手・評価値グラフ・棋譜分析（Phase 7〜）
- **拡張性**: 変則ルール・定跡・LLM 解説も将来対応予定

## 必要環境

- Python 3.12+
- pip

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

NNUE 推論・Web UI も使う場合:

```bash
pip install -e ".[full,dev]"
```

NNUE/KPP の学習も行う場合（M5 Mac では MPS が自動使用される）:

```bash
pip install -e ".[full,train,dev]"
```

以降の操作はすべて venv を有効化した状態（`source .venv/bin/activate`）で行うこと。

## 実行方法

### エンジン起動（USI モード）

```bash
python -m engine
```

### オプション

```bash
python -m engine --search alphabeta --eval material --depth 5
python -m engine --search alphabeta --eval nnue
python -m engine --help
```

### USI コマンドを手動で試す

```bash
printf "usi\nisready\nusinewgame\nposition startpos\ngo\nquit\n" | python -m engine
```

### CLI 盤面ビューア

GUI なしでも、盤面を見ながら数手ぶん AI に指させられる。

```bash
python -m engine.visualize --plies 8
python -m engine.visualize --strategy ranging_rook --plies 8
python -m engine.visualize --book none --depth 3 --plies 4
```

`--clear` を付けると、各手ごとに画面をクリアして表示する。

### ShogiGUI（Windows）から使う

1. エンジンパスに `python` を指定
2. 引数に `-m engine` を指定（プロジェクトのルートディレクトリで実行）

### テスト

```bash
pytest
```

型チェック:

```bash
mypy core/ engine/
```

## プロジェクト構成

```
shogi-ai/
├── core/          # 盤面・手生成・型定義（共通基盤）
├── search/        # 探索アルゴリズム（Alpha-beta, MCTS）
├── eval/          # 評価関数（駒得, KPP, NNUE）
├── engine/        # エンジン本体・USI プロトコル
├── book/          # 定跡・戦法
├── train/         # NNUE / KPP 学習
├── benchmark/     # 詰将棋ベンチマーク・自己対局
├── analysis/      # 棋譜分析・グラフ生成
├── commentary/    # LLM 解説（将来実装）
├── web/           # ブラウザ対局 UI（Phase 7〜）
└── tests/         # テスト
```

詳細は [PLAN.md](PLAN.md) と [STATUS.md](STATUS.md) を参照。
