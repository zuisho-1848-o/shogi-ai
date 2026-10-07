"""テスト用: USI手列 → 棋桜/ShogiGUI風KIF文字列。"""

import shogi

from kifu_review.notation import PIECE_KANJI, ZEN, KAN


def to_kif(usi_moves: list[str], black="テスト先手", white="テスト後手", end="投了") -> str:
    b = shogi.Board()
    lines = ["# KIF version=2.0 encoding=UTF-8", "手合割：平手", f"先手：{black}", f"後手：{white}", "手数----指手---------消費時間--"]
    prev = None
    for i, usi in enumerate(usi_moves, 1):
        mv = shogi.Move.from_usi(usi)
        dest = "同　" if prev == mv.to_square else ZEN[int(shogi.SQUARE_NAMES[mv.to_square][0])] + KAN[ord(shogi.SQUARE_NAMES[mv.to_square][1]) - 96]
        if mv.drop_piece_type:
            body = f"{dest}{PIECE_KANJI[mv.drop_piece_type]}打"
        else:
            pt = b.piece_at(mv.from_square).piece_type
            f, r = shogi.SQUARE_NAMES[mv.from_square]
            body = f"{dest}{PIECE_KANJI[pt]}{'成' if mv.promotion else ''}({f}{ord(r) - 96})"
        lines.append(f"{i:>4} {body}   ( 0:03/00:00:03)")
        b.push(mv)
        prev = mv.to_square
    lines.append(f"{len(usi_moves) + 1:>4} {end}")
    return "\n".join(lines) + "\n"
