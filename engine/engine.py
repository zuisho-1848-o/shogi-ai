from __future__ import annotations

import sys

from book.base import OpeningBook
from book.strategy import Strategy
from core.board import Board, PythonShogiBoard
from core.move_gen import MoveGenerator, PythonShogiMoveGen
from core.types import Move
from engine.config import EngineConfig
from eval.base import Evaluator
from eval.material import MaterialEvaluator
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher
from search.base import SearchResult, Searcher


def _build_evaluator(config: EngineConfig) -> Evaluator:
    if config.eval == "pst":
        return PSTEvaluator()
    if config.eval == "material":
        return MaterialEvaluator()
    # 将来: kpp / nnue
    return MaterialEvaluator()


def _build_searcher(config: EngineConfig) -> Searcher:
    # 将来: mcts
    return AlphaBetaSearcher()


class Engine:
    def __init__(
        self,
        config: EngineConfig,
        book: OpeningBook | None = None,
        strategy: Strategy | None = None,
    ) -> None:
        self._config = config
        self._board: Board = PythonShogiBoard.initial()
        self._move_gen: MoveGenerator = PythonShogiMoveGen()
        self._evaluator: Evaluator = _build_evaluator(config)
        self._searcher: Searcher = _build_searcher(config)
        self._book: OpeningBook | None = book
        self._strategy: Strategy | None = strategy

    def new_game(self) -> None:
        self._board = PythonShogiBoard.initial()

    def set_position(self, tokens: list[str]) -> None:
        if not tokens:
            return
        if tokens[0] == "startpos":
            self._board = PythonShogiBoard.initial()
            move_start = 2 if len(tokens) > 1 and tokens[1] == "moves" else len(tokens)
        elif tokens[0] == "sfen":
            sfen = " ".join(tokens[1:5])
            self._board = PythonShogiBoard.from_sfen(sfen)
            move_start = 6 if len(tokens) > 5 and tokens[5] == "moves" else len(tokens)
        else:
            return
        for usi_move in tokens[move_start:]:
            self._board = self._board.apply_move(Move.from_usi(usi_move))

    def go(self, tokens: list[str]) -> None:
        # 定跡参照（定跡があれば即答）
        if self._book is not None:
            tag = self._strategy.tag if self._strategy else None
            book_move = self._book.lookup(self._board.to_sfen(), tag)
            if book_move is not None:
                print(f"info string book move {book_move.to_usi()}")
                print(f"bestmove {book_move.to_usi()}")
                sys.stdout.flush()
                return

        result: SearchResult = self._searcher.search(
            board=self._board,
            move_gen=self._move_gen,
            evaluator=self._evaluator,
            rules=self._config.rules,
            depth=self._config.depth,
            time_limit_ms=self._config.time_limit_ms,
            multi_pv=self._config.multi_pv,
        )

        for i, cand in enumerate(result.candidates, start=1):
            pv_str = " ".join(m.to_usi() for m in cand.pv) if cand.pv else cand.move.to_usi()
            print(
                f"info depth {result.depth} multipv {i} score cp {cand.score}"
                f" nodes {result.nodes} pv {pv_str}"
            )

        if result.best_move is None:
            print("bestmove resign")
        else:
            print(f"bestmove {result.best_move.to_usi()}")
        sys.stdout.flush()

    def stop(self) -> None:
        self._searcher.stop()

    def set_option(self, name: str, value: str) -> None:
        if name == "MultiPV":
            try:
                self._config.multi_pv = int(value)
            except ValueError:
                pass
