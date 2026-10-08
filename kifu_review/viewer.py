"""解析結果を1枚の自己完結HTML（オフライン可）にする。

盤面は SFEN で持ち、JS 側で描画する。各局面の候補手と読み筋（各手後の局面つき）を埋め込む。
"""

from __future__ import annotations

import json
from pathlib import Path

import shogi

from .analyze import PosAnalysis, winrate
from .kif import KifGame
from .notation import move_to_kanji
from .review import MoveReview

PV_STEPS = 10
TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _pv_steps(board: shogi.Board, pv: list[str], prev_usi: str | None) -> list[dict]:
    """読み筋を1手ずつ進めた [{usi, kanji, sfen}]。不正手で打ち切る。"""
    b = shogi.Board(board.sfen())
    out: list[dict] = []
    prev = prev_usi
    for usi in pv[:PV_STEPS]:
        mv = shogi.Move.from_usi(usi)
        if mv not in b.legal_moves:
            break
        kanji = move_to_kanji(b, usi, prev)
        b.push(mv)
        out.append({"usi": usi, "k": kanji, "sfen": b.sfen()})
        prev = usi
    return out


def build_view_data(game: KifGame, pos: list[PosAnalysis], reviews: list[MoveReview], meta: dict) -> dict:
    board = shogi.Board(game.start_sfen)
    plies = []
    prev: str | None = None
    for i in range(len(game.moves) + 1):
        sign = 1 if board.turn == shogi.BLACK else -1  # 手番側視点 → 先手視点
        cands = []
        for ln in pos[i].lines:
            cands.append(
                {
                    "cp": ln.score_cp * sign, "mate": None if ln.mate is None else ln.mate * sign,
                    "depth": ln.depth, "steps": _pv_steps(board, ln.pv, prev),
                }  # fmt: skip
            )
        cp_s = (pos[i].cp * sign) if pos[i].lines else (-29000 * sign)
        entry = {
            "sfen": board.sfen(), "last": prev, "cp": cp_s, "wr": round(winrate(cp_s), 4),
            "turn": "先手" if board.turn == shogi.BLACK else "後手", "cands": cands,
        }  # fmt: skip
        if i < len(game.moves):
            r = reviews[i]
            entry["move"] = {
                "usi": r.usi, "k": r.kanji, "label": r.label, "mark": r.mark, "loss": round(r.loss_wr, 4),
                "cp_after": r.cp_after * sign, "rank": r.rank, "best": r.best_kanji, "tags": r.tags,
                "sec": r.seconds, "comment": r.comment,
            }  # fmt: skip
            board.push_usi(game.moves[i].usi)
            prev = game.moves[i].usi
        plies.append(entry)
    return {
        "black": game.black_name, "white": game.white_name, "headers": game.headers, "end": game.end_word,
        "focus": meta.get("focus", "両者"), "engine": meta.get("engine", ""), "limit": meta.get("limit", ""),
        "plies": plies,
    }  # fmt: skip


def build_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
