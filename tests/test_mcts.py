"""MCTS テスト: 基本動作・MultiPV・インターフェース準拠・評価関数連携"""
from __future__ import annotations

import math

import pytest

from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color, Move, PieceType, Square
from core.board import Board
from core.move_gen import MoveGenerator
from eval.base import Evaluator
from eval.material import MaterialEvaluator
from search.base import Searcher
from search.mcts import MCTSSearcher, _MCTSNode, _sigmoid


# ------------------------------------------------------------------ helpers

def _search(sfen: str, time_ms: int = 200, multi_pv: int = 1):
    board = PythonShogiBoard.from_sfen(sfen)
    move_gen = PythonShogiMoveGen()
    evaluator = MaterialEvaluator()
    searcher = MCTSSearcher()
    return searcher.search(
        board=board,
        move_gen=move_gen,
        evaluator=evaluator,
        rules=RuleSet(),
        depth=5,
        time_limit_ms=time_ms,
        multi_pv=multi_pv,
    )


_STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


# ------------------------------------------------------------------ unit: _sigmoid

def test_sigmoid_center() -> None:
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_positive() -> None:
    assert _sigmoid(1.0) > 0.5


def test_sigmoid_negative() -> None:
    assert _sigmoid(-1.0) < 0.5


# ------------------------------------------------------------------ unit: _MCTSNode

def test_mctsnode_ucb1_unvisited_is_inf() -> None:
    parent = _MCTSNode(board=None, move=None, parent=None, untried_moves=[], visits=10, total_value=5.0)  # type: ignore[arg-type]
    child = _MCTSNode(board=None, move=None, parent=parent, untried_moves=[], visits=0, total_value=0.0)  # type: ignore[arg-type]
    assert child.ucb1() == float("inf")


def test_mctsnode_ucb1_formula() -> None:
    parent = _MCTSNode(board=None, move=None, parent=None, untried_moves=[], visits=10, total_value=5.0)  # type: ignore[arg-type]
    child = _MCTSNode(board=None, move=None, parent=parent, untried_moves=[], visits=4, total_value=2.0)  # type: ignore[arg-type]
    expected = 2.0 / 4.0 + math.sqrt(2) * math.sqrt(math.log(10) / 4)
    assert abs(child.ucb1() - expected) < 1e-9


def test_mctsnode_is_fully_expanded_true() -> None:
    node = _MCTSNode(board=None, move=None, parent=None, untried_moves=[])  # type: ignore[arg-type]
    assert node.is_fully_expanded()


def test_mctsnode_is_fully_expanded_false() -> None:
    node = _MCTSNode(board=None, move=None, parent=None, untried_moves=[Move.from_usi("7g7f")])
    assert not node.is_fully_expanded()


# ------------------------------------------------------------------ searcher interface

def test_mcts_is_searcher() -> None:
    assert isinstance(MCTSSearcher(), Searcher)


def test_mcts_returns_result_initial() -> None:
    result = _search(_STARTPOS, time_ms=300)
    assert result.best_move is not None
    assert result.nodes > 0


def test_mcts_best_move_is_legal() -> None:
    board = PythonShogiBoard.from_sfen(_STARTPOS)
    move_gen = PythonShogiMoveGen()
    legal_usis = {m.to_usi() for m in move_gen.generate_moves(board, RuleSet())}
    result = _search(_STARTPOS, time_ms=300)
    assert result.best_move is not None
    assert result.best_move.to_usi() in legal_usis


def test_mcts_best_move_matches_first_candidate() -> None:
    result = _search(_STARTPOS, time_ms=300, multi_pv=3)
    assert result.best_move == result.candidates[0].move


def test_mcts_multi_pv_count() -> None:
    result = _search(_STARTPOS, time_ms=300, multi_pv=5)
    assert 1 <= len(result.candidates) <= 5


def test_mcts_candidates_descending_visits() -> None:
    """候補手はスコア（訪問回数に比例）の降順で返る。"""
    result = _search(_STARTPOS, time_ms=300, multi_pv=5)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_mcts_all_candidates_legal() -> None:
    board = PythonShogiBoard.from_sfen(_STARTPOS)
    move_gen = PythonShogiMoveGen()
    legal_usis = {m.to_usi() for m in move_gen.generate_moves(board, RuleSet())}
    result = _search(_STARTPOS, time_ms=300, multi_pv=5)
    for cand in result.candidates:
        assert cand.move.to_usi() in legal_usis


def test_mcts_stop_terminates() -> None:
    """stop() 後に search() を呼んでも正常に終了する。"""
    searcher = MCTSSearcher()
    searcher.stop()
    board = PythonShogiBoard.from_sfen(_STARTPOS)
    result = searcher.search(
        board=board,
        move_gen=PythonShogiMoveGen(),
        evaluator=MaterialEvaluator(),
        rules=RuleSet(),
        depth=5,
        time_limit_ms=1000,
        multi_pv=1,
    )
    assert result.best_move is not None


# ------------------------------------------------------------------ 詰み局面

def test_mcts_captures_free_rook() -> None:
    """飛車タダ取り局面では取り手をほぼ必ず選ぶ（十分な探索時間）。"""
    sfen = "9/9/9/9/6r2/6P2/9/9/9 b - 1"
    result = _search(sfen, time_ms=500)
    assert result.best_move is not None
    assert result.best_move.to_usi() == "3f3e"


# ------------------------------------------------------------------ no legal moves

class _GameOverBoard(Board):
    """合法手なし（即詰み）のスタブボード。"""

    def apply_move(self, move: Move) -> Board:  # pragma: no cover
        return self

    def is_check(self) -> bool:
        return True

    def is_game_over(self) -> bool:
        return True

    @classmethod
    def from_sfen(cls, sfen: str) -> Board:
        return cls()

    def to_sfen(self) -> str:
        return "gameover"

    @property
    def turn(self) -> Color:
        return Color.BLACK

    def piece_at_sq(self, sq: Square) -> tuple[PieceType, Color] | None:
        return None

    def null_move_board(self) -> Board:
        return self


class _EmptyMoveGen(MoveGenerator):
    def generate_moves(self, board: Board, rules: RuleSet) -> list[Move]:
        return []


class _ZeroEval(Evaluator):
    def evaluate(self, board: Board) -> int:
        return 0


def test_mcts_no_legal_moves() -> None:
    """合法手がない局面では best_move = None・candidates 空で返る。"""
    searcher = MCTSSearcher()
    result = searcher.search(
        board=_GameOverBoard(),
        move_gen=_EmptyMoveGen(),
        evaluator=_ZeroEval(),
        rules=RuleSet(),
        depth=5,
        time_limit_ms=200,
        multi_pv=1,
    )
    assert result.best_move is None
    assert result.candidates == []
