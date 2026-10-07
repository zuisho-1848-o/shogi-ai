"""解析結果から『振り返るべき手』を抽出し、悪手の理由につながる事実（タグ）を付ける。"""

from __future__ import annotations

from dataclasses import dataclass, field

import shogi

from .analyze import PosAnalysis, winrate
from .engine import MATE_CP
from .kif import KifGame
from .notation import PIECE_KANJI, material, move_to_kanji, pv_to_kanji, usi_dest

# 勝率の低下幅によるラベル
LABELS = [(0.20, "大悪手", "??"), (0.10, "悪手", "?"), (0.05, "疑問手", "?!")]


def bucket(cp: int) -> str:
    if cp >= 1000:
        return "勝勢"
    if cp >= 400:
        return "優勢"
    if cp >= 150:
        return "やや優勢"
    if cp > -150:
        return "互角"
    if cp > -400:
        return "やや劣勢"
    if cp > -1000:
        return "劣勢"
    return "敗勢"


def phase_of(ply: int) -> str:
    return "序盤" if ply <= 24 else "中盤" if ply <= 80 else "終盤"


@dataclass
class MoveReview:
    ply: int
    side: str  # "先手" / "後手"
    usi: str
    kanji: str
    seconds: int | None
    cp_before: int  # 指す前の評価（指した側視点）
    cp_after: int  # 指した後の評価（指した側視点）
    loss_wr: float  # 勝率低下（0〜1）
    label: str = ""
    mark: str = ""
    rank: int | None = None  # 指した手がMultiPVの何位か（圏外はNone）
    best_usi: str = ""
    best_kanji: str = ""
    best_pv: str = ""
    best_mate: int | None = None
    refutation_pv: str = ""  # 指した手の後、相手の最善進行
    played_pv: str = ""  # 指した手が候補内なら、その読み筋
    tags: list[str] = field(default_factory=list)
    sfen_before: str = ""
    board_before: str = ""
    comment: str = ""

    @property
    def loss_cp(self) -> int:
        return self.cp_before - self.cp_after


def _label(loss: float) -> tuple[str, str]:
    for th, name, mark in LABELS:
        if loss >= th:
            return name, mark
    return "", ""


def _pv_material_delta(board: shogi.Board, pv: list[str], color: int, plies: int = 6) -> int:
    """pv を偶数手ぶん進めたときの『color側の駒得』（開始時との差）。"""
    n = min(len(pv), plies)
    n -= n % 2
    if n == 0:
        return 0
    b = shogi.Board(board.sfen())
    base = material(b, color) - material(b, 1 - color)
    for usi in pv[:n]:
        mv = shogi.Move.from_usi(usi)
        if mv not in b.legal_moves:
            return 0
        b.push(mv)
    return (material(b, color) - material(b, 1 - color)) - base


def review_game(game: KifGame, pos: list[PosAnalysis]) -> list[MoveReview]:
    from .notation import board_text

    board = shogi.Board(game.start_sfen)
    out: list[MoveReview] = []
    prev_usi: str | None = None
    for i, km in enumerate(game.moves):
        before, after = pos[i], pos[i + 1]
        color = board.turn
        side = "先手" if color == shogi.BLACK else "後手"
        mover_best = before.best
        cp_before = before.cp
        played_line = next((l for l in before.lines if l.pv and l.pv[0] == km.usi), None)
        if after.terminal:
            cp_after = MATE_CP  # 相手に合法手なし＝詰ませた
        elif played_line:
            cp_after = played_line.score_cp
        else:
            cp_after = -after.cp
        best_usi = mover_best.pv[0] if mover_best and mover_best.pv else ""
        if best_usi == km.usi:
            cp_after = cp_before
        loss = max(0.0, winrate(cp_before) - winrate(cp_after))
        label, mark = _label(loss)
        mr = MoveReview(
            ply=km.no, side=side, usi=km.usi, kanji=move_to_kanji(board, km.usi, prev_usi),
            seconds=km.seconds, cp_before=cp_before, cp_after=cp_after, loss_wr=loss,
            label=label, mark=mark, rank=played_line.multipv if played_line else None,
            best_usi=best_usi, best_mate=mover_best.mate if mover_best else None,
            sfen_before=board.sfen(), comment=km.comment,
        )  # fmt: skip
        if best_usi:
            mr.best_kanji = move_to_kanji(board, best_usi, prev_usi)
            mr.best_pv = pv_to_kanji(board, mover_best.pv, prev_usi, 10)
        if played_line:
            mr.played_pv = pv_to_kanji(board, played_line.pv, prev_usi, 10)
        if label:  # 重い処理は振り返り対象だけ
            mr.board_before = board_text(board, shogi.SQUARE_NAMES.index(usi_dest(prev_usi)) if prev_usi else None)
            _add_tags(mr, board, before, after, color, km.usi, prev_usi)
        # refutation は着手後局面の最善手順
        b2 = shogi.Board(board.sfen())
        b2.push_usi(km.usi)
        if label and after.best:
            mr.refutation_pv = pv_to_kanji(b2, after.best.pv, km.usi, 10)
        out.append(mr)
        board.push_usi(km.usi)
        prev_usi = km.usi
    return out


def _add_tags(mr: MoveReview, board: shogi.Board, before: PosAnalysis, after: PosAnalysis, color: int, usi: str, prev_usi: str | None) -> None:
    t = mr.tags
    b_before, b_after = bucket(mr.cp_before), bucket(mr.cp_after)
    if b_before != b_after:
        t.append(f"形勢: {b_before}({mr.cp_before:+d}) → {b_after}({mr.cp_after:+d})")
    else:
        t.append(f"形勢: {b_before}のまま（{mr.cp_before:+d} → {mr.cp_after:+d}）")
    mv = shogi.Move.from_usi(usi)
    # 詰み関連
    if mr.best_mate and mr.best_mate > 0 and not (mr.cp_after >= MATE_CP - 100):
        t.append(f"詰みを逃した: {mr.best_mate}手詰があった（最善 {mr.best_kanji}）")
    if after.best and after.best.mate and after.best.mate > 0:
        t.append(f"相手に{after.best.mate}手詰を許した")
    # 指した手の性質
    piece = board.piece_at(mv.from_square) if mv.from_square is not None else None
    cap = board.piece_at(mv.to_square)
    if mv.drop_piece_type:
        t.append(f"駒を打つ手（{PIECE_KANJI[mv.drop_piece_type]}）")
    elif piece and piece.piece_type == shogi.KING:
        t.append("玉を動かす手")
    if cap:
        t.append(f"{PIECE_KANJI[cap.piece_type]}を取る手")
    b2 = shogi.Board(board.sfen())
    b2.push(mv)
    if b2.is_check():
        t.append("王手")
    # 取られる・駒損
    if after.best and after.best.pv:
        first = shogi.Move.from_usi(after.best.pv[0])
        if first.to_square == mv.to_square and b2.piece_at(first.to_square) is not None:
            moved = b2.piece_at(first.to_square)
            t.append(f"動かした駒（{PIECE_KANJI[moved.piece_type]}）をそのまま取られる筋がある")
        d_ref = -_pv_material_delta(b2, after.best.pv, 1 - color)
        if d_ref <= -3:
            t.append(f"相手の最善進行で駒損（約{-d_ref}点分）")
    if before.best and before.best.pv:
        d_best = _pv_material_delta(board, before.best.pv, color)
        if d_best >= 3:
            t.append(f"最善手順は駒得（約{d_best}点分）")
        bm = shogi.Move.from_usi(before.best.pv[0])
        if bm.from_square is not None and board.piece_at(bm.to_square):
            t.append(f"最善は{PIECE_KANJI[board.piece_at(bm.to_square).piece_type]}を取る手")
        bb = shogi.Board(board.sfen())
        bb.push(bm)
        if bb.is_check():
            t.append("最善は王手")
    # 候補順位・時間
    if mr.rank:
        t.append(f"指した手は候補{mr.rank}位（最善と近い手。選び方の差）")
    else:
        t.append("指した手は上位候補に入っていない")
    if mr.seconds is not None:
        if mr.seconds <= 3:
            t.append(f"ほぼ即指し（{mr.seconds}秒）→ 読み不足・手癖の可能性")
        elif mr.seconds >= 120:
            t.append(f"長考（{mr.seconds // 60}分{mr.seconds % 60}秒）")
    t.append(f"局面の段階: {phase_of(mr.ply)}")


def pick_key_moves(reviews: list[MoveReview], side: str | None, top: int) -> list[MoveReview]:
    cands = [r for r in reviews if r.label and (side is None or r.side == side)]
    cands.sort(key=lambda r: r.loss_wr, reverse=True)
    return sorted(cands[:top], key=lambda r: r.ply)


def summary(reviews: list[MoveReview]) -> dict[str, dict]:
    res: dict[str, dict] = {}
    for side in ("先手", "後手"):
        rs = [r for r in reviews if r.side == side]
        cps = [min(max(r.loss_cp, 0), 1000) for r in rs]
        res[side] = {
            "moves": len(rs),
            "avg_loss_cp": round(sum(cps) / len(cps)) if cps else 0,
            "疑問手": sum(1 for r in rs if r.label == "疑問手"),
            "悪手": sum(1 for r in rs if r.label == "悪手"),
            "大悪手": sum(1 for r in rs if r.label == "大悪手"),
            "best_rate": round(100 * sum(1 for r in rs if r.rank == 1 or r.loss_wr == 0) / len(rs)) if rs else 0,
        }
    return res
