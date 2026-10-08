import shogi

from kifu_review.analyze import PosAnalysis
from kifu_review.engine import Line
from kifu_review.kif import KifError, parse_kif
from kifu_review.notation import move_to_kanji
from kifu_review.review import review_game
from tests.kif_writer import to_kif

OPENING = "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b".split()


def test_parse_roundtrip_with_dou_and_promotion():
    moves = "7g7f 3c3d 8h2b+ 3a2b".split()  # 角交換（成り・同）
    g = parse_kif(to_kif(moves))
    assert [m.usi for m in g.moves] == moves
    assert g.end_word == "投了" and g.black_name == "テスト先手"


def test_parse_drop_and_shiftjis():
    b = shogi.Board()
    mv = "7g7f 3c3d 8h2b+ 3a2b".split() + ["B*5e"]
    text = to_kif(mv)
    g = parse_kif(text)
    assert g.moves[-1].usi == "B*5e"
    from kifu_review.kif import _decode

    assert "先手" in _decode(text.encode("cp932"))


def test_illegal_move_rejected():
    text = to_kif(OPENING[:2]).replace("７六歩(77)", "７五歩(77)")
    try:
        parse_kif(text)
    except KifError as e:
        assert "1手目" in str(e)
    else:
        raise AssertionError


def test_kanji_notation():
    b = shogi.Board()
    assert move_to_kanji(b, "7g7f") == "▲７六歩"
    b.push_usi("7g7f")
    assert move_to_kanji(b, "3c3d") == "△３四歩"


def test_review_flags_blunder():
    g = parse_kif(to_kif(OPENING[:2]))
    # 局面0: 最善 7g7f(+0) / 局面1: 後手番で +0 / 局面2: 先手視点で大きく不利(後手+1500)
    pos = [
        PosAnalysis(0, [Line(1, 0, None, 10, ["2g2f"])]),
        PosAnalysis(1, [Line(1, 1500, None, 10, ["3c3d"])]),  # 後手番で+1500 → 先手の ▲7六歩 が悪手扱い
        PosAnalysis(2, [Line(1, 0, None, 10, ["2g2f"])]),
    ]
    rs = review_game(g, pos)
    assert rs[0].label == "大悪手" and rs[0].cp_after == -1500
    assert rs[0].tags  # 理由タグが付く


def test_viewer_data_and_html():
    from kifu_review.viewer import build_html, build_view_data

    g = parse_kif(to_kif(OPENING[:2]))
    pos = [
        PosAnalysis(0, [Line(1, 0, None, 10, ["2g2f", "8c8d"])]),
        PosAnalysis(1, [Line(1, 1500, None, 10, ["3c3d"])]),
        PosAnalysis(2, [Line(1, 0, None, 10, ["2g2f"])]),
    ]
    rs = review_game(g, pos)
    data = build_view_data(g, pos, rs, {"focus": "両者"})
    assert len(data["plies"]) == 3
    assert data["plies"][0]["cands"][0]["steps"][1]["k"] == "△８四歩"
    assert data["plies"][1]["cp"] < 0  # 後手番の+1500 → 先手視点は負
    html = build_html(data)
    assert "/*__DATA__*/" not in html and "△８四歩" in html
