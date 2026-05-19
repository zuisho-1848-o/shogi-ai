"""棋譜の事後分析: 各局面の評価値・候補手を計算する。"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.board import Board, PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color, Move
from eval.base import Evaluator


@dataclass
class MoveAnalysis:
    ply: int
    move: str  # USI 形式
    eval_before: int  # 指す前の評価値（手番側視点）
    eval_after: int  # 指した後の評価値（先手視点）
    candidates: list[dict] = field(default_factory=list)  # 指す前の候補手リスト


def analyze_game(
    moves: list[str],
    evaluator: Evaluator,
    *,
    depth: int = 0,
    multi_pv: int = 3,
) -> list[MoveAnalysis]:
    """USI 手順を事後分析し、各手の評価値と候補手を返す。

    depth=0 は evaluator のみ（高速）。depth>0 は AlphaBeta 探索（低速・正確）。
    """
    board: Board = PythonShogiBoard.initial()
    move_gen = PythonShogiMoveGen()
    rules = RuleSet()
    results: list[MoveAnalysis] = []

    searcher = None
    if depth > 0:
        from search.alphabeta import AlphaBetaSearcher
        searcher = AlphaBetaSearcher()

    for ply, usi_move in enumerate(moves, start=1):
        eval_before = evaluator.evaluate(board)

        candidates: list[dict] = []
        if searcher is not None:
            result = searcher.search(
                board=board,
                move_gen=move_gen,
                evaluator=evaluator,
                rules=rules,
                depth=depth,
                time_limit_ms=5_000,
                multi_pv=multi_pv,
            )
            candidates = [
                {"rank": i + 1, "move": c.move.to_usi(), "score": c.score}
                for i, c in enumerate(result.candidates)
            ]

        board = board.apply_move(Move.from_usi(usi_move))

        # 先手視点の評価値を記録
        eval_after = evaluator.evaluate(board)
        if board.turn == Color.WHITE:
            # 後手番 → evaluate は後手視点 → 符号反転して先手視点に
            eval_after = -eval_after

        results.append(
            MoveAnalysis(
                ply=ply,
                move=usi_move,
                eval_before=eval_before,
                eval_after=eval_after,
                candidates=candidates,
            )
        )

    return results
