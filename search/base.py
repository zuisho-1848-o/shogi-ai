from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.board import Board
from core.move_gen import MoveGenerator
from core.rules import RuleSet
from core.types import Move
from eval.base import Evaluator


@dataclass
class CandidateMove:
    move: Move
    score: int
    pv: list[Move] = field(default_factory=list)


@dataclass
class SearchResult:
    best_move: Move | None
    best_score: int
    candidates: list[CandidateMove]
    depth: int
    nodes: int


class Searcher(ABC):
    @abstractmethod
    def search(
        self,
        board: Board,
        move_gen: MoveGenerator,
        evaluator: Evaluator,
        rules: RuleSet,
        depth: int,
        time_limit_ms: int,
        multi_pv: int,
    ) -> SearchResult: ...

    def stop(self) -> None:
        pass
