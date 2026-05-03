from core.board import PythonShogiBoard
from core.types import Color, Move


def test_initial_board_turn() -> None:
    board = PythonShogiBoard.initial()
    assert board.turn == Color.BLACK


def test_initial_board_sfen_contains_pieces() -> None:
    board = PythonShogiBoard.initial()
    sfen = board.to_sfen()
    assert "lnsgkgsnl" in sfen


def test_from_sfen_initial_position() -> None:
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    board = PythonShogiBoard.from_sfen(sfen)
    assert board.turn == Color.BLACK


def test_apply_move_changes_turn() -> None:
    board = PythonShogiBoard.initial()
    new_board = board.apply_move(Move.from_usi("7g7f"))
    assert new_board.turn == Color.WHITE


def test_apply_move_is_immutable() -> None:
    board = PythonShogiBoard.initial()
    _ = board.apply_move(Move.from_usi("7g7f"))
    assert board.turn == Color.BLACK  # 元の盤面は変わらない


def test_is_check_false_at_initial() -> None:
    board = PythonShogiBoard.initial()
    assert not board.is_check()


def test_is_game_over_false_at_initial() -> None:
    board = PythonShogiBoard.initial()
    assert not board.is_game_over()
