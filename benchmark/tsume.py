"""詰将棋ベンチマーク。AlphaBetaSearcher + PST で組み込み問題を解いて強さを測定する。

使い方:
    python -m benchmark.tsume
    python -m benchmark.tsume --depth-bonus 2 --time-limit 15000
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher

_MATE_THRESHOLD = 8_000_000  # この値以上のスコアを詰みと判定


@dataclass
class TsumeProblem:
    sfen: str
    mate_in: int         # 詰み手数 (先手の手数)
    description: str
    expected_move: str | None = None   # 正解初手 USI (None = 複数解あり)


@dataclass
class TsumeResult:
    problem: TsumeProblem
    solved: bool
    found_move: str | None
    elapsed_ms: float
    nodes: int


# 組み込み問題集 (verified: AlphaBeta で解いて正解を確認済み)
DEFAULT_PROBLEMS: list[TsumeProblem] = [
    TsumeProblem(
        # 盤面: 先手銀 3a, 後手玉 1a  /  先手: 金持ち
        # 詰み: G*2b → 1b(金カバー), 2a(金カバー), 2b(銀でカバー) に逃げられず
        # G*2b のみが即詰み (G*1b は玉が香を取れる, G*2a は玉が 1b に逃げられる)
        sfen="6S1k/9/9/9/9/9/9/9/9 b G 1",
        mate_in=1,
        description="1手詰め: 角隅金打(唯一解)",
        expected_move="G*2b",
    ),
]


class TsumeBenchmark:
    """詰将棋ベンチマーク。

    Args:
        problems: テスト問題リスト (None で DEFAULT_PROBLEMS を使用)
        time_limit_ms: 1問あたりの探索時間上限
        depth_bonus: 詰み手数 × 2 + 1 に加算する追加深さ
    """

    def __init__(
        self,
        problems: list[TsumeProblem] | None = None,
        time_limit_ms: int = 10_000,
        depth_bonus: int = 0,
    ) -> None:
        self._problems = problems if problems is not None else DEFAULT_PROBLEMS
        self._time_limit_ms = time_limit_ms
        self._depth_bonus = depth_bonus

    def run_problem(self, problem: TsumeProblem) -> TsumeResult:
        board = PythonShogiBoard.from_sfen(problem.sfen)
        searcher = AlphaBetaSearcher()

        # 詰みは奇数手なので depth = 2×N+1+bonus
        depth = problem.mate_in * 2 + 1 + self._depth_bonus

        t0 = time.time()
        result = searcher.search(
            board=board,
            move_gen=PythonShogiMoveGen(),
            evaluator=PSTEvaluator(),
            rules=RuleSet(),
            depth=depth,
            time_limit_ms=self._time_limit_ms,
            multi_pv=1,
        )
        elapsed_ms = (time.time() - t0) * 1000

        solved = bool(result.best_move and result.best_score >= _MATE_THRESHOLD)
        found = result.best_move.to_usi() if result.best_move else None

        return TsumeResult(
            problem=problem,
            solved=solved,
            found_move=found,
            elapsed_ms=elapsed_ms,
            nodes=result.nodes,
        )

    def run(self) -> list[TsumeResult]:
        return [self.run_problem(p) for p in self._problems]

    @staticmethod
    def report(results: list[TsumeResult]) -> str:
        lines = ["=== 詰将棋ベンチマーク ==="]
        solved_count = 0
        for r in results:
            status = "✓" if r.solved else "✗"
            move = r.found_move or "-"
            exp = r.problem.expected_move or "不定"
            lines.append(
                f"  {status} [{r.problem.mate_in}手詰め] {r.problem.description}"
                f" | 手:{move} 期待:{exp}"
                f" | {r.elapsed_ms:.0f}ms / {r.nodes:,}nodes"
            )
            if r.solved:
                solved_count += 1
        lines.append(f"\n正解: {solved_count}/{len(results)}")
        return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="詰将棋ベンチマーク")
    parser.add_argument("--time-limit", type=int, default=10_000)
    parser.add_argument("--depth-bonus", type=int, default=0)
    args = parser.parse_args()

    bench = TsumeBenchmark(
        time_limit_ms=args.time_limit,
        depth_bonus=args.depth_bonus,
    )
    results = bench.run()
    print(TsumeBenchmark.report(results))


if __name__ == "__main__":
    main()
