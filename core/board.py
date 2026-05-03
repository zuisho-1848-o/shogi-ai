from __future__ import annotations

from abc import ABC, abstractmethod

import shogi

from core.types import Color, Move, PieceType, Square

INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


class Board(ABC):
    """盤面の抽象インターフェース。apply_move は新インスタンスを返す（イミュータブル API）。"""

    @abstractmethod
    def apply_move(self, move: Move) -> Board: ...

    @abstractmethod
    def is_check(self) -> bool: ...

    @abstractmethod
    def is_game_over(self) -> bool: ...

    @classmethod
    @abstractmethod
    def from_sfen(cls, sfen: str) -> Board: ...

    @abstractmethod
    def to_sfen(self) -> str: ...

    @property
    @abstractmethod
    def turn(self) -> Color: ...

    @abstractmethod
    def piece_at_sq(self, sq: Square) -> tuple[PieceType, Color] | None:
        """指定マスの駒種と色を返す。空きマスは None。"""
        ...

    @abstractmethod
    def null_move_board(self) -> Board:
        """手番を反転させた盤面を返す（Null Move Pruning 用）。"""
        ...


class PythonShogiBoard(Board):
    """python-shogi の shogi.Board をラップした Board 実装。
    将来 NativeBoard に差し替え可能。直接依存はこのファイルにのみ許可。
    """

    def __init__(self, shogi_board: shogi.Board) -> None:
        self._board = shogi_board

    @classmethod
    def initial(cls) -> PythonShogiBoard:
        return cls(shogi.Board())

    @classmethod
    def from_sfen(cls, sfen: str) -> PythonShogiBoard:
        return cls(shogi.Board(sfen))

    def apply_move(self, move: Move) -> PythonShogiBoard:
        # shogi.Board に copy() がないため SFEN 経由でコピーする
        new_board = shogi.Board(self._board.sfen())
        new_board.push_usi(move.to_usi())
        return PythonShogiBoard(new_board)

    def is_check(self) -> bool:
        return self._board.is_check()

    def is_game_over(self) -> bool:
        return self._board.is_game_over()

    def to_sfen(self) -> str:
        return self._board.sfen()

    @property
    def turn(self) -> Color:
        return Color(self._board.turn)

    def piece_at_sq(self, sq: Square) -> tuple[PieceType, Color] | None:
        piece = self._board.piece_at(sq)
        if piece is None:
            return None
        try:
            return PieceType(piece.piece_type), Color(piece.color)
        except ValueError:
            return None

    def null_move_board(self) -> PythonShogiBoard:
        parts = self._board.sfen().split()
        parts[1] = "w" if parts[1] == "b" else "b"
        return PythonShogiBoard.from_sfen(" ".join(parts))

    def get_shogi_board(self) -> shogi.Board:
        """python-shogi 内部オブジェクトへのアクセス。MoveGenerator 専用。"""
        return self._board
