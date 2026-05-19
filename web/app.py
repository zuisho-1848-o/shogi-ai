"""将棋AI Web API (FastAPI) — Phase 7: WebSocket / MultiPV / 評価値グラフ"""
from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import shogi
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.result import format_candidates
from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher

app = FastAPI(title="Shogi AI")

PIECE_KANJI: dict[int, str] = {
    shogi.PAWN: "歩", shogi.LANCE: "香", shogi.KNIGHT: "桂",
    shogi.SILVER: "銀", shogi.GOLD: "金", shogi.BISHOP: "角",
    shogi.ROOK: "飛", shogi.KING: "王", shogi.PROM_PAWN: "と",
    shogi.PROM_LANCE: "杏", shogi.PROM_KNIGHT: "圭",
    shogi.PROM_SILVER: "全", shogi.PROM_BISHOP: "馬", shogi.PROM_ROOK: "龍",
}

PIECE_NAMES: dict[int, str] = {
    shogi.PAWN: "pawn", shogi.LANCE: "lance", shogi.KNIGHT: "knight",
    shogi.SILVER: "silver", shogi.GOLD: "gold", shogi.BISHOP: "bishop",
    shogi.ROOK: "rook", shogi.KING: "king", shogi.PROM_PAWN: "prom_pawn",
    shogi.PROM_LANCE: "prom_lance", shogi.PROM_KNIGHT: "prom_knight",
    shogi.PROM_SILVER: "prom_silver", shogi.PROM_BISHOP: "prom_bishop",
    shogi.PROM_ROOK: "prom_rook",
}

HAND_ORDER = [
    shogi.ROOK, shogi.BISHOP, shogi.GOLD,
    shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN,
]

# ------------------------------------------------------------------ global state

_board = shogi.Board()
_move_history: list[str] = []
_eval_history: list[int] = []   # 先手視点の評価値（手ごと）
_last_candidates: list[dict] = []

# AI コンポーネント（起動時に1度だけ初期化）
_evaluator = PSTEvaluator()
_searcher = AlphaBetaSearcher()
_move_gen = PythonShogiMoveGen()
_rules = RuleSet()
_executor = ThreadPoolExecutor(max_workers=1)


# ------------------------------------------------------------------ WebSocket

class _WSManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(data)
            except Exception:
                dead.append(client)
        for ws in dead:
            self.disconnect(ws)


_ws_manager = _WSManager()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await _ws_manager.connect(ws)
    try:
        await ws.send_json(_extended_state())
        while True:
            await ws.receive_text()  # keep-alive (ping)
    except WebSocketDisconnect:
        _ws_manager.disconnect(ws)
    except Exception:
        _ws_manager.disconnect(ws)


# ------------------------------------------------------------------ helpers

def _black_eval() -> int:
    """現在の盤面を先手視点の評価値に変換する。"""
    if _board.is_game_over():
        return -9_000_000 if _board.turn == shogi.BLACK else 9_000_000
    wrapper = PythonShogiBoard(_board)
    score = _evaluator.evaluate(wrapper)
    # evaluate() は手番側視点 → 後手番なら反転して先手視点に揃える
    if _board.turn == shogi.WHITE:
        score = -score
    return score


def _board_state() -> dict:
    pieces = []
    for sq in range(81):
        p = _board.piece_at(sq)
        if p is None:
            continue
        file = 9 - (sq % 9)
        rank = sq // 9 + 1
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


def _extended_state() -> dict:
    state = _board_state()
    state["eval_history"] = list(_eval_history)
    state["last_candidates"] = list(_last_candidates)
    return state


# ------------------------------------------------------------------ endpoints

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/new-game")
async def new_game() -> dict:
    global _board, _move_history, _eval_history, _last_candidates
    _board = shogi.Board()
    _move_history = []
    _eval_history = []
    _last_candidates = []
    state = _extended_state()
    await _ws_manager.broadcast(state)
    return state


@app.get("/api/state")
async def get_state() -> dict:
    return _extended_state()


class MoveRequest(BaseModel):
    move: str  # USI 形式 例: "7g7f", "P*5e"


@app.post("/api/move")
async def make_move(req: MoveRequest) -> dict:
    global _board, _move_history, _eval_history
    legal = {m.usi() for m in _board.legal_moves}
    if req.move not in legal:
        raise HTTPException(status_code=400, detail=f"Illegal move: {req.move}")
    _board.push_usi(req.move)
    _move_history.append(req.move)
    _eval_history.append(_black_eval())
    state = _extended_state()
    await _ws_manager.broadcast(state)
    return state


class AIMoveRequest(BaseModel):
    depth: int = 3


@app.post("/api/ai-move")
async def ai_move(req: AIMoveRequest = AIMoveRequest()) -> dict:
    global _board, _move_history, _eval_history, _last_candidates
    if _board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    board_wrapper = PythonShogiBoard(_board)
    depth = max(1, min(req.depth, 7))

    # ブロッキング探索をスレッドプールで実行（イベントループを止めない）
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _searcher.search(
            board=board_wrapper,
            move_gen=_move_gen,
            evaluator=_evaluator,
            rules=_rules,
            depth=depth,
            time_limit_ms=10_000,
            multi_pv=5,
        ),
    )

    _last_candidates = format_candidates(result, max_n=5)

    if result.best_move is None:
        state = _extended_state()
        state.update({"resigned": True, "winner": "black", "is_game_over": True})
        await _ws_manager.broadcast(state)
        return state

    move_usi = result.best_move.to_usi()
    _board.push_usi(move_usi)
    _move_history.append(move_usi)
    _eval_history.append(_black_eval())

    state = _extended_state()
    state["ai_move"] = move_usi
    state["ai_score"] = result.best_score
    await _ws_manager.broadcast(state)
    return state
