"""将棋AI Web API (FastAPI)"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import shogi
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher

app = FastAPI(title="Shogi AI")

PIECE_KANJI: dict[int, str] = {
    shogi.PAWN: "歩",
    shogi.LANCE: "香",
    shogi.KNIGHT: "桂",
    shogi.SILVER: "銀",
    shogi.GOLD: "金",
    shogi.BISHOP: "角",
    shogi.ROOK: "飛",
    shogi.KING: "王",
    shogi.PROM_PAWN: "と",
    shogi.PROM_LANCE: "杏",
    shogi.PROM_KNIGHT: "圭",
    shogi.PROM_SILVER: "全",
    shogi.PROM_BISHOP: "馬",
    shogi.PROM_ROOK: "龍",
}

PIECE_NAMES: dict[int, str] = {
    shogi.PAWN: "pawn",
    shogi.LANCE: "lance",
    shogi.KNIGHT: "knight",
    shogi.SILVER: "silver",
    shogi.GOLD: "gold",
    shogi.BISHOP: "bishop",
    shogi.ROOK: "rook",
    shogi.KING: "king",
    shogi.PROM_PAWN: "prom_pawn",
    shogi.PROM_LANCE: "prom_lance",
    shogi.PROM_KNIGHT: "prom_knight",
    shogi.PROM_SILVER: "prom_silver",
    shogi.PROM_BISHOP: "prom_bishop",
    shogi.PROM_ROOK: "prom_rook",
}

HAND_ORDER = [
    shogi.ROOK, shogi.BISHOP, shogi.GOLD,
    shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN,
]

# Global game state
_board = shogi.Board()
_move_history: list[str] = []

# AI components (lazy-initialized once)
_evaluator = PSTEvaluator()
_searcher = AlphaBetaSearcher()
_move_gen = PythonShogiMoveGen()
_rules = RuleSet()


def _board_state() -> dict:
    pieces = []
    for sq in range(81):
        p = _board.piece_at(sq)
        if p is None:
            continue
        file = 9 - (sq % 9)  # 1-9 (9=left, 1=right from black's view)
        rank = sq // 9 + 1   # 1-9 (1=top, 9=bottom from black's view)
        pieces.append({
            "sq": sq,
            "file": file,
            "rank": rank,
            "piece": PIECE_NAMES.get(p.piece_type, "unknown"),
            "kanji": PIECE_KANJI.get(p.piece_type, "?"),
            "color": "black" if p.color == shogi.BLACK else "white",
            "promoted": p.piece_type >= shogi.PROM_PAWN,
        })

    hands: dict[str, dict[str, int]] = {"black": {}, "white": {}}
    for pt in HAND_ORDER:
        name = PIECE_NAMES[pt]
        hands["black"][name] = _board.pieces_in_hand[shogi.BLACK].get(pt, 0)
        hands["white"][name] = _board.pieces_in_hand[shogi.WHITE].get(pt, 0)

    legal_moves = [m.usi() for m in _board.legal_moves]

    game_over = _board.is_game_over()
    winner: Optional[str] = None
    if game_over:
        winner = "white" if _board.turn == shogi.BLACK else "black"

    return {
        "turn": "black" if _board.turn == shogi.BLACK else "white",
        "pieces": pieces,
        "hands": hands,
        "legal_moves": legal_moves,
        "is_game_over": game_over,
        "winner": winner,
        "last_move": _move_history[-1] if _move_history else None,
        "move_count": len(_move_history),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/new-game")
async def new_game():
    global _board, _move_history
    _board = shogi.Board()
    _move_history = []
    return _board_state()


@app.get("/api/state")
async def get_state():
    return _board_state()


class MoveRequest(BaseModel):
    move: str  # USI format e.g. "7g7f", "P*5e"


@app.post("/api/move")
async def make_move(req: MoveRequest):
    global _board, _move_history
    legal = {m.usi() for m in _board.legal_moves}
    if req.move not in legal:
        raise HTTPException(status_code=400, detail=f"Illegal move: {req.move}")
    _board.push_usi(req.move)
    _move_history.append(req.move)
    return _board_state()


class AIMoveRequest(BaseModel):
    depth: int = 3


@app.post("/api/ai-move")
async def ai_move(req: AIMoveRequest = AIMoveRequest()):
    global _board, _move_history
    if _board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    board_wrapper = PythonShogiBoard(_board)
    result = _searcher.search(
        board=board_wrapper,
        move_gen=_move_gen,
        evaluator=_evaluator,
        rules=_rules,
        depth=max(1, min(req.depth, 7)),
        time_limit_ms=10_000,
        multi_pv=1,
    )

    if result.best_move is None:
        state = _board_state()
        state["resigned"] = True
        state["winner"] = "black"
        state["is_game_over"] = True
        return state

    move_usi = result.best_move.to_usi()
    _board.push_usi(move_usi)
    _move_history.append(move_usi)

    state = _board_state()
    state["ai_move"] = move_usi
    state["ai_score"] = result.best_score
    return state
