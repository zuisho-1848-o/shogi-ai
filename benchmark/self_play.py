"""自己対局ベンチマーク。2つの評価関数を対戦させてレーティングを比較する。

使い方:
    python -m benchmark.self_play                     # PST vs NNUE (デフォルト)
    python -m benchmark.self_play --n-games 20 --depth 3
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color
from eval.base import Evaluator
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher

_RESIGN_THRESHOLD = 8_000_000  # この絶対値以上のスコアで投了


@dataclass
class SelfPlayConfig:
    depth: int = 3
    time_limit_ms: int = 1_000
    max_moves: int = 400     # この手数で引き分け


@dataclass
class SelfPlayResult:
    n_games: int
    engine1_wins: int
    engine2_wins: int
    draws: int
    elapsed_s: float

    @property
    def engine1_win_rate(self) -> float:
        return self.engine1_wins / self.n_games if self.n_games else 0.0

    def __str__(self) -> str:
        wr = self.engine1_win_rate * 100
        return (
            f"Engine1: {self.engine1_wins}勝 / Engine2: {self.engine2_wins}勝 / "
            f"引き分け: {self.draws} (計{self.n_games}局, "
            f"Engine1勝率 {wr:.1f}%, {self.elapsed_s:.1f}s)"
        )


class SelfPlay:
    """2つの評価関数を同一の AlphaBeta で対戦させる。

    Args:
        eval1: Engine1 の評価関数
        eval2: Engine2 の評価関数
        config: 探索設定
    """

    def __init__(
        self,
        eval1: Evaluator,
        eval2: Evaluator,
        config: SelfPlayConfig | None = None,
    ) -> None:
        self._eval1 = eval1
        self._eval2 = eval2
        self._cfg = config or SelfPlayConfig()

    def _play_game(self, engine1_is_black: bool) -> int:
        """1局対戦。戻り値: 1=Engine1勝, -1=Engine2勝, 0=引き分け。"""
        board = PythonShogiBoard.initial()
        move_gen = PythonShogiMoveGen()
        rules = RuleSet()
        # 各プレイヤーに独立した探索器 (TT・killers を共有しない)
        searcher1 = AlphaBetaSearcher()
        searcher2 = AlphaBetaSearcher()

        for _ in range(self._cfg.max_moves):
            is_black_turn = board.turn == Color.BLACK
            engine1_turn = is_black_turn == engine1_is_black

            searcher = searcher1 if engine1_turn else searcher2
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

            # 手がない or 大差で投了
            if result.best_move is None or result.best_score <= -_RESIGN_THRESHOLD:
                # 現在の手番側が投了 → 相手の勝ち
                opponent_is_black = not is_black_turn
                engine1_won = opponent_is_black == engine1_is_black
                return 1 if engine1_won else -1

            board = board.apply_move(result.best_move)

            if board.is_game_over():
                # 手を指した後、相手が詰んでいる → 指した側の勝ち
                mover_was_black = not (board.turn == Color.BLACK)
                engine1_won = mover_was_black == engine1_is_black
                return 1 if engine1_won else -1

        return 0  # 最大手数に達したら引き分け

    def run(self, n_games: int = 10) -> SelfPlayResult:
        wins = losses = draws = 0
        t0 = time.time()

        for i in range(n_games):
            # 交互に先後を入れ替える
            outcome = self._play_game(engine1_is_black=(i % 2 == 0))
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

        return SelfPlayResult(
            n_games=n_games,
            engine1_wins=wins,
            engine2_wins=losses,
            draws=draws,
            elapsed_s=time.time() - t0,
        )


def main() -> None:
    import argparse
    from pathlib import Path
    from eval.nnue import NNUEEvaluator

    parser = argparse.ArgumentParser(description="自己対局ベンチマーク")
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-limit-ms", type=int, default=1_000)
    args = parser.parse_args()

    eval1 = PSTEvaluator()
    eval2 = NNUEEvaluator()
    label2 = "NNUE" if eval2.has_model else "NNUE(→PST fallback)"

    cfg = SelfPlayConfig(
        depth=args.depth,
        time_limit_ms=args.time_limit_ms,
    )
    print(f"PST vs {label2} | {args.n_games}局 depth={args.depth}")
    sp = SelfPlay(eval1, eval2, cfg)
    result = sp.run(args.n_games)
    print(f"  Engine1(PST): {result.engine1_wins}勝")
    print(f"  Engine2({label2}): {result.engine2_wins}勝")
    print(f"  引き分け: {result.draws}")
    print(f"  Engine1 勝率: {result.engine1_win_rate*100:.1f}%")
    print(f"  経過時間: {result.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
