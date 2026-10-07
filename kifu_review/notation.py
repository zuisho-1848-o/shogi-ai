"""python-shogi の局面 → 日本語表記（▲７六歩 等）と盤面テキスト。"""

from __future__ import annotations

import shogi

ZEN = "０１２３４５６７８９"
KAN = "〇一二三四五六七八九"
PIECE_KANJI = {
    shogi.PAWN: "歩", shogi.LANCE: "香", shogi.KNIGHT: "桂", shogi.SILVER: "銀",
    shogi.GOLD: "金", shogi.BISHOP: "角", shogi.ROOK: "飛", shogi.KING: "玉",
    shogi.PROM_PAWN: "と", shogi.PROM_LANCE: "成香", shogi.PROM_KNIGHT: "成桂",
    shogi.PROM_SILVER: "成銀", shogi.PROM_BISHOP: "馬", shogi.PROM_ROOK: "龍",
}  # fmt: skip
# 盤面表示用の1文字（成り駒は1字）
BOARD_CHAR = {**PIECE_KANJI, shogi.PROM_LANCE: "杏", shogi.PROM_KNIGHT: "圭", shogi.PROM_SILVER: "全"}
# 駒の価値（歩=1 基準の目安。駒損の概算用）
PIECE_VALUE = {
    shogi.PAWN: 1, shogi.LANCE: 3, shogi.KNIGHT: 4, shogi.SILVER: 5, shogi.GOLD: 6,
    shogi.BISHOP: 8, shogi.ROOK: 10, shogi.KING: 0, shogi.PROM_PAWN: 6,
    shogi.PROM_LANCE: 6, shogi.PROM_KNIGHT: 6, shogi.PROM_SILVER: 6,
    shogi.PROM_BISHOP: 10, shogi.PROM_ROOK: 12,
}  # fmt: skip
HAND_ORDER = [shogi.ROOK, shogi.BISHOP, shogi.GOLD, shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN]


def _sq_name(sq: int) -> str:
    return shogi.SQUARE_NAMES[sq]  # '7f'


def _fmt_sq(name: str) -> str:
    return f"{ZEN[int(name[0])]}{KAN[ord(name[1]) - ord('a') + 1]}"


def _promo_zone(color: int, sq: int) -> bool:
    rank = sq // 9  # 0=一段目
    return rank <= 2 if color == shogi.BLACK else rank >= 6


def _usi_dest(usi: str) -> str:
    return usi[2:4]  # 通常手 '7g7f'・成り '8h2b+'・打ち 'P*7f' いずれも 2〜3文字目が移動先


usi_dest = _usi_dest


def move_to_kanji(board: shogi.Board, usi: str, prev_usi: str | None = None) -> str:
    """合法手 usi を ▲７六歩 形式に。曖昧な場合は (77) を付与。board は着手前の局面。"""
    move = shogi.Move.from_usi(usi)
    mark = "▲" if board.turn == shogi.BLACK else "△"
    dest = _fmt_sq(_sq_name(move.to_square))
    if prev_usi and _usi_dest(prev_usi) == usi_dest(usi):
        dest = "同　"  # 直前の手の移動先と同じ
    if move.drop_piece_type:
        # 打：同じ駒種が盤上から同地点へ動ける場合のみ『打』を付ける
        same = any(
            m.from_square is not None and not m.drop_piece_type and m.to_square == move.to_square
            and board.piece_at(m.from_square).piece_type == move.drop_piece_type
            for m in board.legal_moves
        )
        return f"{mark}{dest}{PIECE_KANJI[move.drop_piece_type]}{'打' if same else ''}"
    piece = board.piece_at(move.from_square)
    name = PIECE_KANJI[piece.piece_type]
    suffix = ""
    if move.promotion:
        suffix = "成"
    elif piece.piece_type in (shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.BISHOP, shogi.ROOK) and (
        _promo_zone(board.turn, move.from_square) or _promo_zone(board.turn, move.to_square)
    ):
        suffix = "不成"
    ambiguous = any(
        m.to_square == move.to_square and m.from_square != move.from_square and m.from_square is not None
        and board.piece_at(m.from_square).piece_type == piece.piece_type
        for m in board.legal_moves
    )
    if ambiguous:
        f, r = _sq_name(move.from_square)
        suffix += f"({f}{ord(r) - ord('a') + 1})"
    return f"{mark}{dest}{name}{suffix}"


def pv_to_kanji(board: shogi.Board, pv: list[str], prev_usi: str | None = None, limit: int = 10) -> str:
    """PV（USI手のリスト）を日本語に。局面は破壊しない。不正手に当たれば打ち切る。"""
    b = board.copy() if hasattr(board, "copy") else shogi.Board(board.sfen())
    out: list[str] = []
    prev = prev_usi
    for usi in pv[:limit]:
        try:
            move = shogi.Move.from_usi(usi)
            if move not in b.legal_moves:
                break
            out.append(move_to_kanji(b, usi, prev))
            b.push(move)
            prev = usi
        except Exception:
            break
    return " ".join(out)


def material(board: shogi.Board, color: int) -> int:
    """color 側の駒価値合計（盤上＋持ち駒）。"""
    total = 0
    for sq in range(81):
        p = board.piece_at(sq)
        if p and p.color == color:
            total += PIECE_VALUE[p.piece_type]
    for pt, n in board.pieces_in_hand[color].items():
        total += PIECE_VALUE[pt] * n
    return total


def _hand_text(board: shogi.Board, color: int) -> str:
    h = board.pieces_in_hand[color]
    parts = []
    for pt in HAND_ORDER:
        n = h.get(pt, 0)
        if n:
            parts.append(PIECE_KANJI[pt] + (KAN[n] if n > 1 else ""))
    return "".join(parts) or "なし"


def board_text(board: shogi.Board, last_to: int | None = None) -> str:
    """盤面を等幅テキストで。後手の駒は『v』付き、直前の着手先は『*』で強調。"""
    lines = [f"後手の持駒：{_hand_text(board, shogi.WHITE)}", " " + "".join(f" {ZEN[n]} " for n in range(9, 0, -1)), "+" + "-" * 36 + "+"]
    for rank in range(9):
        cells = []
        for file_idx in range(9):
            sq = rank * 9 + file_idx
            p = board.piece_at(sq)
            if p is None:
                cell = " ・ "
            else:
                pre = "v" if p.color == shogi.WHITE else " "
                cell = pre + BOARD_CHAR[p.piece_type] + ("*" if sq == last_to else " ")
            cells.append(cell)
        lines.append("|" + "".join(cells) + f"|{KAN[rank + 1]}")
    lines.append("+" + "-" * 36 + "+")
    lines.append(f"先手の持駒：{_hand_text(board, shogi.BLACK)}")
    return "\n".join(lines)
