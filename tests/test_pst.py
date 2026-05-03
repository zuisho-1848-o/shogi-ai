"""PST 評価関数のテスト"""
from core.board import PythonShogiBoard
from eval.pst import PSTEvaluator


def test_pst_initial_position_zero() -> None:
    """初期局面は両者均等なので 0 になる。"""
    board = PythonShogiBoard.initial()
    ev = PSTEvaluator()
    assert ev.evaluate(board) == 0


def test_pst_after_pawn_advance_nonzero() -> None:
    """歩を進めた後は PST ボーナスで非対称になる（手番が変わるので絶対値が非 0）。"""
    board = PythonShogiBoard.initial()
    from core.types import Move
    board = board.apply_move(Move.from_usi("7g7f"))
    ev = PSTEvaluator()
    score = ev.evaluate(board)
    # 手番は後手になるので符号が変わる。初期局面より変化しているはず
    assert score != 0  # 先手が歩を進めたので非対称


def test_pst_returns_int() -> None:
    """evaluate は int を返す。"""
    board = PythonShogiBoard.initial()
    ev = PSTEvaluator()
    score = ev.evaluate(board)
    assert isinstance(score, int)


def test_pst_with_alphabeta() -> None:
    """PST を使った探索が正常に動作する。"""
    from core.move_gen import PythonShogiMoveGen
    from core.rules import RuleSet
    from search.alphabeta import AlphaBetaSearcher

    board = PythonShogiBoard.initial()
    searcher = AlphaBetaSearcher()
    result = searcher.search(
        board=board,
        move_gen=PythonShogiMoveGen(),
        evaluator=PSTEvaluator(),
        rules=RuleSet(),
        depth=2,
        time_limit_ms=5000,
        multi_pv=3,
    )
    assert result.best_move is not None
    assert len(result.candidates) >= 1


def test_pst_capture_position() -> None:
    """駒取り局面で PST 評価も駒得を正しく反映する。"""
    from core.board import PythonShogiBoard
    from core.move_gen import PythonShogiMoveGen
    from core.rules import RuleSet
    from search.alphabeta import AlphaBetaSearcher

    # 3f の歩が 3e の飛車を取れる局面
    sfen = "9/9/9/9/6r2/6P2/9/9/9 b - 1"
    board = PythonShogiBoard.from_sfen(sfen)
    searcher = AlphaBetaSearcher()
    result = searcher.search(
        board=board,
        move_gen=PythonShogiMoveGen(),
        evaluator=PSTEvaluator(),
        rules=RuleSet(),
        depth=2,
        time_limit_ms=5000,
        multi_pv=1,
    )
    assert result.best_move is not None
    assert result.best_move.to_usi() == "3f3e"
