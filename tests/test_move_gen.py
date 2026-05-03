from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Move


def test_initial_position_has_30_moves() -> None:
    """初期局面の合法手は 30 手。"""
    board = PythonShogiBoard.initial()
    gen = PythonShogiMoveGen()
    moves = gen.generate_moves(board, RuleSet())
    assert len(moves) == 30


def test_moves_not_empty() -> None:
    board = PythonShogiBoard.initial()
    gen = PythonShogiMoveGen()
    moves = gen.generate_moves(board, RuleSet())
    assert len(moves) > 0


def test_all_moves_have_valid_usi() -> None:
    board = PythonShogiBoard.initial()
    gen = PythonShogiMoveGen()
    moves = gen.generate_moves(board, RuleSet())
    for move in moves:
        usi = move.to_usi()
        assert len(usi) >= 4


def test_move_usi_round_trip() -> None:
    """Move の USI 変換は往復で一致する。"""
    cases = ["7g7f", "2g2f", "8h2b+", "P*3d"]
    for usi in cases:
        assert Move.from_usi(usi).to_usi() == usi


def test_moves_after_first_move() -> None:
    """1手目後も合法手が存在する。"""
    board = PythonShogiBoard.initial()
    board = board.apply_move(Move.from_usi("7g7f"))
    gen = PythonShogiMoveGen()
    moves = gen.generate_moves(board, RuleSet())
    assert len(moves) > 0
