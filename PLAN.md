# アーキテクチャ設計

---

## 設計の大原則

**「まず動かす、後から差し替える」**

各コンポーネントは ABC（抽象基底クラス）で定義し、実装を後から差し替えられる設計。

| コンポーネント | 現在の実装 | 差し替え候補 |
|---|---|---|
| 盤面・手生成 | `python-shogi` ラッパー | 自作合法手生成エンジン（NativeBoard） |
| 評価関数 | PST / KPP / NNUE | 独自評価、Hi!story 用評価 |
| 探索 | Alpha-beta / MCTS | その他手法 |
| ルールセット | 標準将棋 | 変則ルール、Hi!story ルール |
| 解説生成 | なし | LLM（Claude API） |

---

## モジュール依存の方向

```
core（types, board, move_gen, rules）
    ↓
eval（base, material, pst, kpp, nnue）
search（base, alphabeta, mcts, tt）
    ↓
engine（engine, config, usi）
    ↓
benchmark / analysis / web

train（dataset, kpp_train）← core + eval のみ依存（独立）
```

循環依存なし。`core` が最上流。

---

## コーディング規約

| 規約 | 内容 |
|---|---|
| 型ヒント | 全関数・メソッドに必須（mypy strict 適合） |
| 依存方向 | 上記の方向を厳守。逆方向の import 禁止 |
| グローバル状態 | 持たない。`Board.apply_move()` は新インスタンスを返す |
| `python-shogi` 依存 | `core/board.py` と `core/move_gen.py` だけに閉じ込める |
| パス | `pathlib.Path` を使う（ハードコードの `/` 区切り禁止） |
| stdout 出力 | `print()` のみ（USI プロトコルは LF、`\r\n` 混入禁止） |

---

## 将来の拡張計画

### LLM 解説機能

`commentary/base.py` に `Commentator` ABC を置き、実装を後から追加する。

```python
class Commentator(ABC):
    @abstractmethod
    def comment(self, board: Board, result: SearchResult) -> str: ...

# commentary/llm.py（未実装）
class ClaudeCommentator(Commentator):
    """Claude API を使って局面の解説文を生成"""
```

### 変則ルール対応

`RuleSet` にフラグを追加するだけで対応できる。`MoveGenerator` の実装を変えなくてよい。

```python
config = EngineConfig(
    rules=RuleSet(
        allow_double_pawn=True,             # 二歩あり
        king_moves_only_when_in_check=True, # 取れる時しか玉を動かせない
        allow_arbitrary_start=True,         # 任意の初期配置
    )
)
```

### Hi!story AI への応用

| 将棋AIの技術 | Hi!story AIへの転用 |
|---|---|
| `Board` ABC + `RuleSet` | Hi!story 独自ルールの盤面・アクション管理 |
| Alpha-beta | カードゲームの先読み |
| **MCTS** | **Hi!story ルールへの汎化（最も有力）** |
| KPP / NNUE | 対戦データからの自動評価学習 |
| Web UI | ブラウザ上でのデジタル対戦 |
| LLM 解説 | カード効果・戦略の解説 |
