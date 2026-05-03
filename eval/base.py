from __future__ import annotations

from abc import ABC, abstractmethod

from core.board import Board


class Evaluator(ABC):
    """局面評価の抽象インターフェース。返り値は手番側から見た centipawn スコア。"""

    @abstractmethod
    def evaluate(self, board: Board) -> int: ...
