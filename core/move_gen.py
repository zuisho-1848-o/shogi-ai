from __future__ import annotations

from abc import ABC, abstractmethod

import shogi

from core.board import Board, PythonShogiBoard
from core.rules import RuleSet
from core.types import Move


class MoveGenerator(ABC):
    """合法手生成の抽象インターフェース。"""

    @abstractmethod
    def generate_moves(self, board: Board, rules: RuleSet) -> list[Move]: ...


class PythonShogiMoveGen(MoveGenerator):
    """python-shogi による合法手生成。変則ルールは RuleSet に従って追加フィルタリング。"""

    def generate_moves(self, board: Board, rules: RuleSet) -> list[Move]:
        assert isinstance(board, PythonShogiBoard)
        shogi_board = board.get_shogi_board()
        moves = [Move.from_usi(m.usi()) for m in shogi_board.legal_moves]
        # 変則ルールのフィルタリングは Phase 3 で実装
        return moves
