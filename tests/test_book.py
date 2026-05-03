"""定跡・戦法のテスト"""
from book.base import OpeningBook
from book.sfen_book import SfenBook
from book.strategy import STRATEGY_MAP, Strategy


_INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_normalize_sfen_removes_movenum() -> None:
    normalized = OpeningBook.normalize_sfen(_INITIAL_SFEN)
    parts = normalized.split()
    assert len(parts) == 3  # board turn hands (手数なし)


def test_minimal_book_lookup_initial() -> None:
    """最小定跡から初期局面の手が返る。"""
    book = SfenBook.minimal()
    move = book.lookup(_INITIAL_SFEN)
    assert move is not None
    assert move.to_usi() in {"7g7f", "2g2f", "6g6f"}


def test_minimal_book_lookup_unknown_position() -> None:
    """未知局面では None を返す。"""
    book = SfenBook.minimal()
    unknown = "9/9/9/9/9/9/9/9/9 b - 1"
    assert book.lookup(unknown) is None


def test_book_strategy_filter_static_rook() -> None:
    """居飛車タグ付きの手が優先される。"""
    book = SfenBook.minimal()
    move = book.lookup(_INITIAL_SFEN, strategy_tag="static_rook")
    assert move is not None
    # 居飛車の手（7g7f, 2g2f）が返るはず
    assert move.to_usi() in {"7g7f", "2g2f"}


def test_book_strategy_filter_ranging_rook() -> None:
    """振り飛車タグ付きの手が優先される。"""
    book = SfenBook.minimal()
    move = book.lookup(_INITIAL_SFEN, strategy_tag="ranging_rook")
    assert move is not None
    # 振り飛車の手 (7g7f or 6g6f) が返るはず
    assert move.to_usi() in {"7g7f", "6g6f"}


def test_strategy_map_contains_presets() -> None:
    """STRATEGY_MAP に主要戦法が登録されている。"""
    assert "free" in STRATEGY_MAP
    assert "ranging_rook" in STRATEGY_MAP
    assert "static_rook" in STRATEGY_MAP


def test_strategy_tag() -> None:
    """Strategy の tag が正しく設定されている。"""
    s = STRATEGY_MAP["ranging_rook"]
    assert s.tag == "ranging_rook"
    assert STRATEGY_MAP["free"].tag is None


def test_book_from_file(tmp_path) -> None:
    """ファイルから定跡を読み込める。"""
    book_content = """# test book
sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -
7g7f 100 static_rook
2g2f 80 static_rook
"""
    book_file = tmp_path / "test.sfen"
    book_file.write_text(book_content, encoding="utf-8")

    book = SfenBook.from_file(book_file)
    move = book.lookup(_INITIAL_SFEN)
    assert move is not None
    assert move.to_usi() == "7g7f"
