"""自己対局ベンチマーク。2つのエンジン設定を対戦させてレーティングを比較する。

使い方:
    python -m benchmark.self_play                                        # PST vs NNUE (デフォルト)
    python -m benchmark.self_play --eval1 pst --eval2 kpp --n-games 50
    python -m benchmark.self_play --eval1 nnue --search1 mcts --eval2 pst --n-games 20
    python -m benchmark.self_play --eval1 pst --eval2 pst --search1 alphabeta --search2 mcts
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color
from eval.base import Evaluator
from eval.kpp import KPPEvaluator
from eval.material import MaterialEvaluator
from eval.nnue import NNUEEvaluator
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher
from search.base import Searcher
from search.mcts import MCTSSearcher

_RESIGN_THRESHOLD = 8_000_000

EVAL_CHOICES = ["pst", "material", "nnue", "kpp"]
SEARCH_CHOICES = ["alphabeta", "mcts"]


def build_evaluator(name: str) -> Evaluator:
    if name == "material":
        return MaterialEvaluator()
    if name == "nnue":
        return NNUEEvaluator()
    if name == "kpp":
        return KPPEvaluator.load_or_fallback()
    return PSTEvaluator()


def build_searcher(name: str) -> Searcher:
    if name == "mcts":
        return MCTSSearcher()
    return AlphaBetaSearcher()


@dataclass
class SelfPlayConfig:
    depth: int = 3
    time_limit_ms: int = 1_000
    max_moves: int = 400


@dataclass
class SelfPlayResult:
    n_games: int
    engine1_wins: int
    engine2_wins: int
    draws: int
    elapsed_s: float
    label1: str = "Engine1"
    label2: str = "Engine2"

    @property
    def engine1_win_rate(self) -> float:
        return self.engine1_wins / self.n_games if self.n_games else 0.0

    def __str__(self) -> str:
        wr = self.engine1_win_rate * 100
        return (
            f"{self.label1}: {self.engine1_wins}勝 / "
            f"{self.label2}: {self.engine2_wins}勝 / "
            f"引き分け: {self.draws} (計{self.n_games}局, "
            f"{self.label1} 勝率 {wr:.1f}%, {self.elapsed_s:.1f}s)"
        )


class SelfPlay:
    """2つのエンジン設定を対戦させる。

    Args:
        eval1, search1: Engine1 の評価関数・探索器
        eval2, search2: Engine2 の評価関数・探索器
        config: 探索設定（depth / time_limit_ms はどちらのエンジンにも適用）
    """

    def __init__(
        self,
        eval1: Evaluator,
        eval2: Evaluator,
        config: SelfPlayConfig | None = None,
        search1: Searcher | None = None,
        search2: Searcher | None = None,
    ) -> None:
        self._eval1 = eval1
        self._eval2 = eval2
        self._search1 = search1 or AlphaBetaSearcher()
        self._search2 = search2 or AlphaBetaSearcher()
        self._cfg = config or SelfPlayConfig()

    def _play_game(self, engine1_is_black: bool) -> int:
        """1局対戦。戻り値: 1=Engine1勝, -1=Engine2勝, 0=引き分け。"""
        board = PythonShogiBoard.initial()
        move_gen = PythonShogiMoveGen()
        rules = RuleSet()

        for _ in range(self._cfg.max_moves):
            is_black_turn = board.turn == Color.BLACK
            engine1_turn = is_black_turn == engine1_is_black

            searcher = self._search1 if engine1_turn else self._search2
            evaluator = self._eval1 if engine1_turn else self._eval2

            result = searcher.search(
                board=board,
                move_gen=move_gen,
                evaluator=evaluator,
                rules=rules,
                depth=self._cfg.depth,
                time_limit_ms=self._cfg.time_limit_ms,
                multi_pv=1,
            )

            if result.best_move is None or result.best_score <= -_RESIGN_THRESHOLD:
                opponent_is_black = not is_black_turn
                engine1_won = opponent_is_black == engine1_is_black
                return 1 if engine1_won else -1

            board = board.apply_move(result.best_move)

            if board.is_game_over():
                mover_was_black = not (board.turn == Color.BLACK)
                engine1_won = mover_was_black == engine1_is_black
                return 1 if engine1_won else -1

        return 0

    def run(self, n_games: int = 10, verbose: bool = True) -> SelfPlayResult:
        wins = losses = draws = 0
        t0 = time.time()

        for i in range(n_games):
            outcome = self._play_game(engine1_is_black=(i % 2 == 0))
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1
            if verbose:
                print(f"  局{i+1:3d}/{n_games}: {'Engine1勝' if outcome > 0 else 'Engine2勝' if outcome < 0 else '引き分け'}"
                      f"  (通算 {wins}勝{losses}敗{draws}分)", flush=True)

        return SelfPlayResult(
            n_games=n_games,
            engine1_wins=wins,
            engine2_wins=losses,
            draws=draws,
            elapsed_s=time.time() - t0,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="自己対局ベンチマーク")
    parser.add_argument("--eval1", choices=EVAL_CHOICES, default="pst", help="Engine1の評価関数")
    parser.add_argument("--eval2", choices=EVAL_CHOICES, default="nnue", help="Engine2の評価関数")
    parser.add_argument("--search1", choices=SEARCH_CHOICES, default="alphabeta", help="Engine1の探索")
    parser.add_argument("--search2", choices=SEARCH_CHOICES, default="alphabeta", help="Engine2の探索")
    parser.add_argument("--n-games", type=int, default=10, help="対局数")
    parser.add_argument("--depth", type=int, default=3, help="探索深さ")
    parser.add_argument("--time-limit-ms", type=int, default=1_000, help="1手あたりの時間制限(ms)")
    parser.add_argument("--quiet", action="store_true", help="局ごとの進捗を非表示")
    args = parser.parse_args()

    eval1 = build_evaluator(args.eval1)
    eval2 = build_evaluator(args.eval2)
    search1 = build_searcher(args.search1)
    search2 = build_searcher(args.search2)

    label1 = f"{args.eval1}+{args.search1}"
    label2 = f"{args.eval2}+{args.search2}"

    cfg = SelfPlayConfig(depth=args.depth, time_limit_ms=args.time_limit_ms)

    print(f"{label1} vs {label2} | {args.n_games}局 depth={args.depth} time={args.time_limit_ms}ms")
    sp = SelfPlay(eval1, eval2, cfg, search1=search1, search2=search2)
    result = sp.run(args.n_games, verbose=not args.quiet)
    result.label1 = label1
    result.label2 = label2

    print()
    print("=== 結果 ===")
    print(f"  {label1}: {result.engine1_wins}勝")
    print(f"  {label2}: {result.engine2_wins}勝")
    print(f"  引き分け: {result.draws}")
    print(f"  {label1} 勝率: {result.engine1_win_rate*100:.1f}%")
    print(f"  経過時間: {result.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
