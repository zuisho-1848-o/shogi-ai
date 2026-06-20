"""SFEN + 候補手 → LLMプロンプト用テキストへの変換"""
from __future__ import annotations

from dataclasses import dataclass

import shogi

PIECE_KANJI: dict[int, str] = {
    shogi.PAWN: "歩", shogi.LANCE: "香", shogi.KNIGHT: "桂",
    shogi.SILVER: "銀", shogi.GOLD: "金", shogi.BISHOP: "角",
    shogi.ROOK: "飛", shogi.KING: "玉", shogi.PROM_PAWN: "と",
    shogi.PROM_LANCE: "杏", shogi.PROM_KNIGHT: "圭",
    shogi.PROM_SILVER: "全", shogi.PROM_BISHOP: "馬", shogi.PROM_ROOK: "龍",
}

RANK_KANJI = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]

HAND_ORDER = [
    shogi.ROOK, shogi.BISHOP, shogi.GOLD,
    shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN,
]

_DROP_CHAR: dict[str, int] = {
    "P": shogi.PAWN, "L": shogi.LANCE, "N": shogi.KNIGHT,
    "S": shogi.SILVER, "G": shogi.GOLD, "B": shogi.BISHOP, "R": shogi.ROOK,
}
_RANK_CHAR_TO_KANJI = {chr(ord("a") + i): RANK_KANJI[i] for i in range(9)}


@dataclass
class BoardContext:
    board_text: str
    hands_text: str
    turn_text: str
    move_count: int
    phase: str
    is_check: bool
    eval_text: str
    candidates_text: str


def _sq_to_jp(sq_str: str) -> str:
    """USI座標（例: 7f）→ 日本語表記（例: ７六）"""
    return f"{sq_str[0]}{_RANK_CHAR_TO_KANJI.get(sq_str[1], sq_str[1])}"


def _usi_move_to_jp(move_usi: str, board: shogi.Board) -> str:
    """USI手（例: 7g7f）→ 日本語手表記（例: ７六歩）"""
    if len(move_usi) < 4:
        return move_usi
    if "*" in move_usi:
        pt = _DROP_CHAR.get(move_usi[0].upper())
        kanji = PIECE_KANJI.get(pt, move_usi[0]) if pt else move_usi[0]
        return f"{_sq_to_jp(move_usi[2:4])}{kanji}打"
    from_file = int(move_usi[0])
    from_rank = ord(move_usi[1]) - ord("a") + 1
    from_sq = (9 - from_file) + (from_rank - 1) * 9
    piece = board.piece_at(from_sq)
    piece_kanji = PIECE_KANJI.get(piece.piece_type, "?") if piece else "?"
    promote = len(move_usi) > 4 and move_usi[4] == "+"
    return f"{_sq_to_jp(move_usi[2:4])}{piece_kanji}{'成' if promote else ''}"


def _board_to_text(board: shogi.Board) -> str:
    lines = ["  ９ ８ ７ ６ ５ ４ ３ ２ １"]
    for rank in range(1, 10):
        cells = []
        for file in range(9, 0, -1):
            sq = (9 - file) + (rank - 1) * 9
            piece = board.piece_at(sq)
            if piece is None:
                cells.append("・")
            else:
                kanji = PIECE_KANJI.get(piece.piece_type, "？")
                prefix = "v" if piece.color == shogi.WHITE else " "
                cells.append(f"{prefix}{kanji}")
        lines.append(f"{RANK_KANJI[rank - 1]}|{''.join(cells)}|")
    return "\n".join(lines)


def _hands_to_text(board: shogi.Board) -> str:
    def _hand(color: int, label: str) -> str:
        pieces = []
        for pt in HAND_ORDER:
            n = board.pieces_in_hand[color].get(pt, 0)
            if n > 0:
                kanji = PIECE_KANJI[pt]
                pieces.append(f"{kanji}×{n}" if n > 1 else kanji)
        return f"{label}: {'　'.join(pieces) if pieces else 'なし'}"

    return _hand(shogi.WHITE, "後手持ち駒") + "\n" + _hand(shogi.BLACK, "先手持ち駒")


def _phase(move_count: int, top_score: int | None) -> str:
    if top_score is not None and abs(top_score) >= 2000:
        return "終盤（勝負どころ）"
    if move_count <= 30:
        return "序盤"
    if move_count <= 80:
        return "中盤"
    return "終盤"


def _eval_to_text(score: int | None) -> str:
    if score is None:
        return "評価値: 不明"
    abs_s = abs(score)
    side = "先手" if score > 0 else "後手"
    if abs_s >= 3000:
        return f"評価値: {score:+d}cp（{side}の勝勢）"
    if abs_s >= 1000:
        return f"評価値: {score:+d}cp（{side}が優勢）"
    if abs_s >= 300:
        return f"評価値: {score:+d}cp（{side}がやや有利）"
    return f"評価値: {score:+d}cp（ほぼ互角）"


def build_context(
    sfen: str,
    candidates: list[dict],
    move_count: int = 0,
) -> BoardContext:
    board = shogi.Board(sfen)
    is_black = board.turn == shogi.BLACK
    turn_text = "先手（▲）番" if is_black else "後手（△）番"
    top_score: int | None = candidates[0]["score"] if candidates else None

    cand_lines = []
    for i, c in enumerate(candidates[:5], 1):
        jp = _usi_move_to_jp(c.get("move", ""), board)
        score = c.get("score", 0)
        pv = c.get("pv", [])
        pv_str = (" 読み筋: " + " ".join(pv[:3])) if pv else ""
        cand_lines.append(f"  {i}位: {jp}（{score:+d}cp）{pv_str}")

    return BoardContext(
        board_text=_board_to_text(board),
        hands_text=_hands_to_text(board),
        turn_text=turn_text,
        move_count=move_count,
        phase=_phase(move_count, top_score),
        is_check=board.is_check(),
        eval_text=_eval_to_text(top_score),
        candidates_text="\n".join(cand_lines) if cand_lines else "  候補手なし",
    )
