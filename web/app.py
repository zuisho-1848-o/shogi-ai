"""将棋AI Web API (FastAPI) — マルチエンジン・AI vs AI 対応版"""
from __future__ import annotations

import asyncio
import copy
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
from web.usi_engine import USIEngine

app = FastAPI(title="Shogi AI")

# ------------------------------------------------------------------ エンジンカタログ

INTERNAL_ENGINES: dict[str, dict] = {
    "pst_alphabeta": {
        "id": "pst_alphabeta",
        "name": "PST + αβ探索",
        "description": "駒得評価 + Alpha-Beta探索（標準）",
        "type": "internal",
    },
    "kpp_alphabeta": {
        "id": "kpp_alphabeta",
        "name": "KPP + αβ探索",
        "description": "KP評価（機械学習済み） + Alpha-Beta探索",
        "type": "internal",
    },
    "nnue_alphabeta": {
        "id": "nnue_alphabeta",
        "name": "NNUE + αβ探索",
        "description": "ニューラルネット評価 + Alpha-Beta探索",
        "type": "internal",
    },
    "pst_mcts": {
        "id": "pst_mcts",
        "name": "PST + MCTS",
        "description": "駒得評価 + モンテカルロ木探索",
        "type": "internal",
    },
}

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

# ------------------------------------------------------------------ グローバル状態

_board = shogi.Board()
_move_history: list[str] = []
_eval_history: list[int] = []
_last_candidates: list[dict] = []
_last_candidates_turn: str = "black"

# エンジンキャッシュ（評価関数・探索器）
_engine_cache: dict[str, tuple] = {}

# プレイヤー設定
class PlayerConfig(BaseModel):
    type: str = "human"           # human | internal | usi
    engine_id: str = "pst_alphabeta"
    depth: int = 3
    time_limit_ms: int = 5000
    usi_path: str = ""

_black_config = PlayerConfig(type="human")
_white_config = PlayerConfig(type="internal", engine_id="pst_alphabeta", depth=3)

# USI エンジンインスタンス
_usi_black: USIEngine | None = None
_usi_white: USIEngine | None = None

_move_gen = PythonShogiMoveGen()
_rules = RuleSet()
_executor = ThreadPoolExecutor(max_workers=1)
_analysis_task: asyncio.Task | None = None


# ------------------------------------------------------------------ エンジンファクトリ

def _get_internal_engine(engine_id: str) -> tuple:
    """評価関数と探索器のペアを返す（キャッシュあり）。"""
    if engine_id in _engine_cache:
        return _engine_cache[engine_id]

    if engine_id == "kpp_alphabeta":
        try:
            import numpy as np
            from eval.kpp import KPPEvaluator
            model_path = Path("models/kpp.npz")
            if model_path.exists():
                data = np.load(model_path)
                ev = KPPEvaluator(data["table"])
            else:
                ev = PSTEvaluator()
        except Exception:
            ev = PSTEvaluator()
        sr = AlphaBetaSearcher()
    elif engine_id == "nnue_alphabeta":
        try:
            from eval.nnue import NNUEEvaluator
            ev = NNUEEvaluator(Path("models/nnue.npz"))
        except Exception:
            ev = PSTEvaluator()
        sr = AlphaBetaSearcher()
    elif engine_id == "pst_mcts":
        from search.mcts import MCTSSearcher
        ev = PSTEvaluator()
        sr = MCTSSearcher()
    else:  # pst_alphabeta (デフォルト)
        ev = PSTEvaluator()
        sr = AlphaBetaSearcher()

    _engine_cache[engine_id] = (ev, sr)
    return ev, sr


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
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_manager.disconnect(ws)
    except Exception:
        _ws_manager.disconnect(ws)


# ------------------------------------------------------------------ ヘルパー

def _black_eval() -> int:
    if _board.is_game_over():
        return -9_000_000 if _board.turn == shogi.BLACK else 9_000_000
    config = _black_config if _board.turn == shogi.BLACK else _white_config
    if config.type == "usi":
        evaluator = PSTEvaluator()
    else:
        evaluator, _ = _get_internal_engine(config.engine_id)
    wrapper = PythonShogiBoard(_board)
    score = evaluator.evaluate(wrapper)
    if _board.turn == shogi.WHITE:
        score = -score
    return score


_DROP_LETTER_TO_TYPE: dict[str, int] = {
    "P": shogi.PAWN, "L": shogi.LANCE, "N": shogi.KNIGHT,
    "S": shogi.SILVER, "G": shogi.GOLD, "B": shogi.BISHOP, "R": shogi.ROOK,
}


def _annotate_piece_kanji(candidates: list[dict], board: shogi.Board) -> None:
    """候補手リストに駒漢字を付加する（着手前の局面で呼ぶこと）。"""
    for cand in candidates:
        move = cand.get("move", "")
        if len(move) >= 2 and move[1] == "*":
            pt = _DROP_LETTER_TO_TYPE.get(move[0].upper())
            cand["piece_kanji"] = PIECE_KANJI.get(pt, "") if pt else ""
            continue
        if len(move) < 4:
            cand["piece_kanji"] = ""
            continue
        try:
            mv = shogi.Move.from_usi(move)
            piece = board.piece_at(mv.from_square)
            cand["piece_kanji"] = PIECE_KANJI.get(piece.piece_type, "") if piece else ""
        except Exception:
            cand["piece_kanji"] = ""


def _player_info(config: PlayerConfig) -> dict:
    if config.type == "human":
        return {"type": "human", "display_name": "人間", "icon": "👤"}
    elif config.type == "usi":
        name = Path(config.usi_path).name if config.usi_path else "未設定"
        return {
            "type": "usi",
            "display_name": f"USI: {name}",
            "icon": "🔌",
            "usi_path": config.usi_path,
            "time_limit_ms": config.time_limit_ms,
        }
    else:
        eng = INTERNAL_ENGINES.get(config.engine_id, {})
        return {
            "type": "internal",
            "engine_id": config.engine_id,
            "display_name": eng.get("name", config.engine_id),
            "icon": "🤖",
            "depth": config.depth,
            "time_limit_ms": config.time_limit_ms,
        }


def _game_mode() -> str:
    bh = _black_config.type == "human"
    wh = _white_config.type == "human"
    if bh and not wh:
        return "human_vs_ai"
    if not bh and wh:
        return "ai_vs_human"
    if not bh and not wh:
        return "ai_vs_ai"
    return "human_vs_human"


def _board_state() -> dict:
    pieces = []
    for sq in range(81):
        p = _board.piece_at(sq)
        if p is None:
            continue
        file = 9 - (sq % 9)
        rank = sq // 9 + 1
        pieces.append({
            "sq": sq, "file": file, "rank": rank,
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
    state["last_candidates_turn"] = _last_candidates_turn
    state["black_player"] = _player_info(_black_config)
    state["white_player"] = _player_info(_white_config)
    state["mode"] = _game_mode()
    return state


async def _schedule_analysis(depth: int = 3) -> None:
    global _analysis_task, _last_candidates

    if _analysis_task and not _analysis_task.done():
        _analysis_task.cancel()

    async def _run() -> None:
        global _last_candidates, _last_candidates_turn
        if _board.is_game_over():
            return
        board_snap = copy.deepcopy(_board)
        is_black_turn = board_snap.turn == shogi.BLACK
        config = _black_config if is_black_turn else _white_config

        if config.type == "usi":
            return  # USIエンジン側が探索済み

        evaluator, searcher = _get_internal_engine(config.engine_id)
        board_wrapper = PythonShogiBoard(board_snap)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: searcher.search(
                    board=board_wrapper,
                    move_gen=_move_gen,
                    evaluator=evaluator,
                    rules=_rules,
                    depth=depth,
                    time_limit_ms=8_000,
                    multi_pv=8,
                ),
            )
        except asyncio.CancelledError:
            return
        _last_candidates = format_candidates(
            result, max_n=8, black_turn=is_black_turn
        )
        _annotate_piece_kanji(_last_candidates, board_snap)
        _last_candidates_turn = "black" if is_black_turn else "white"
        await _ws_manager.broadcast(_extended_state())

    _analysis_task = asyncio.create_task(_run())


# ------------------------------------------------------------------ エンドポイント

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/engines")
async def list_engines() -> dict:
    """利用可能なエンジン一覧を返す。"""
    return {"internal": list(INTERNAL_ENGINES.values())}


class SetupRequest(BaseModel):
    black: PlayerConfig
    white: PlayerConfig


@app.post("/api/setup")
async def setup_players(req: SetupRequest) -> dict:
    """先手・後手のプレイヤーを設定する。既存USIエンジンはシャットダウン。"""
    global _black_config, _white_config, _usi_black, _usi_white

    # 旧USIエンジンを終了
    if _usi_black:
        await _usi_black.quit()
        _usi_black = None
    if _usi_white:
        await _usi_white.quit()
        _usi_white = None

    _black_config = req.black
    _white_config = req.white

    # 新しいUSIエンジンを起動
    errors: list[str] = []
    if req.black.type == "usi":
        if not req.black.usi_path or not Path(req.black.usi_path).exists():
            errors.append(f"先手のUSIエンジンが見つかりません: {req.black.usi_path}")
        else:
            try:
                _usi_black = USIEngine(req.black.usi_path)
                await _usi_black.start()
            except Exception as e:
                errors.append(f"先手USIエンジン起動失敗: {e}")
                _usi_black = None

    if req.white.type == "usi":
        if not req.white.usi_path or not Path(req.white.usi_path).exists():
            errors.append(f"後手のUSIエンジンが見つかりません: {req.white.usi_path}")
        else:
            try:
                _usi_white = USIEngine(req.white.usi_path)
                await _usi_white.start()
            except Exception as e:
                errors.append(f"後手USIエンジン起動失敗: {e}")
                _usi_white = None

    state = _extended_state()
    if errors:
        state["errors"] = errors
    return state


@app.post("/api/new-game")
async def new_game() -> dict:
    global _board, _move_history, _eval_history, _last_candidates

    _board = shogi.Board()
    _move_history = []
    _eval_history = []
    _last_candidates = []

    # USIエンジンに新局を通知
    if _usi_black and _usi_black.is_running:
        await _usi_black.new_game()
    if _usi_white and _usi_white.is_running:
        await _usi_white.new_game()

    state = _extended_state()
    await _ws_manager.broadcast(state)
    await _schedule_analysis(depth=3)
    return state


@app.get("/api/state")
async def get_state() -> dict:
    return _extended_state()


class SetPositionRequest(BaseModel):
    sfen: str


@app.post("/api/set-position")
async def set_position(req: SetPositionRequest) -> dict:
    """任意の SFEN 局面をセットして対局を開始する。"""
    global _board, _move_history, _eval_history, _last_candidates
    try:
        _board = shogi.Board(req.sfen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SFEN: {e}")
    _move_history = []
    _eval_history = []
    _last_candidates = []

    if _usi_black and _usi_black.is_running:
        await _usi_black.new_game()
    if _usi_white and _usi_white.is_running:
        await _usi_white.new_game()

    state = _extended_state()
    await _ws_manager.broadcast(state)
    await _schedule_analysis(depth=3)
    return state


class MoveRequest(BaseModel):
    move: str


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
    await _schedule_analysis(depth=3)
    return state


@app.post("/api/ai-move")
async def ai_move() -> dict:
    """現在の手番プレイヤー（AIのみ）が1手指す。"""
    global _board, _move_history, _eval_history, _last_candidates, _last_candidates_turn

    if _board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    is_black_turn = _board.turn == shogi.BLACK
    config = _black_config if is_black_turn else _white_config

    if config.type == "human":
        raise HTTPException(status_code=400, detail="Current player is human")

    # 進行中のバックグラウンド解析をキャンセル
    global _analysis_task
    if _analysis_task and not _analysis_task.done():
        _analysis_task.cancel()

    # ── USI エンジン ──
    if config.type == "usi":
        usi = _usi_black if is_black_turn else _usi_white
        if usi is None or not usi.is_running:
            raise HTTPException(status_code=503, detail="USI engine not running")

        result = await usi.search(
            moves=list(_move_history),
            time_limit_ms=config.time_limit_ms,
            multi_pv=8,
        )
        move_usi = result.best_move
        _last_candidates = result.candidates
        _annotate_piece_kanji(_last_candidates, _board)
        _last_candidates_turn = "black" if is_black_turn else "white"

        if move_usi is None:
            state = _extended_state()
            state.update({"resigned": True, "is_game_over": True,
                          "winner": "white" if is_black_turn else "black"})
            await _ws_manager.broadcast(state)
            return state

        _board.push_usi(move_usi)
        _move_history.append(move_usi)
        _eval_history.append(_black_eval())
        state = _extended_state()
        state["ai_move"] = move_usi
        state["ai_score"] = result.score
        await _ws_manager.broadcast(state)
        return state

    # ── 内部エンジン ──
    evaluator, searcher = _get_internal_engine(config.engine_id)
    depth = max(1, min(config.depth, 7))
    board_snap = copy.deepcopy(_board)
    board_wrapper = PythonShogiBoard(board_snap)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: searcher.search(
            board=board_wrapper,
            move_gen=_move_gen,
            evaluator=evaluator,
            rules=_rules,
            depth=depth,
            time_limit_ms=10_000,
            multi_pv=8,
        ),
    )

    _last_candidates = format_candidates(result, max_n=8, black_turn=is_black_turn)
    _annotate_piece_kanji(_last_candidates, _board)
    _last_candidates_turn = "black" if is_black_turn else "white"

    if result.best_move is None:
        state = _extended_state()
        state.update({"resigned": True, "is_game_over": True,
                      "winner": "white" if is_black_turn else "black"})
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

    # 着手後のバックグラウンド解析
    await _schedule_analysis(depth=depth)
    return state
