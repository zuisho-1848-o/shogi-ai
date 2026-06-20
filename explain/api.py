"""盤面解説APIルーター — POST /explain"""
from __future__ import annotations

import os
from typing import Optional

import shogi
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from explain.commentator import Commentator, make_backend
from explain.context_builder import build_context

router = APIRouter()

_DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
_VALID_LEVELS = {"beginner", "intermediate", "advanced"}
_VALID_BACKENDS = {"ollama", "claude"}


class CandidateMove(BaseModel):
    move: str
    score: int = 0
    pv: list[str] = []
    rank: int = 0
    piece_kanji: str = ""


class ExplainRequest(BaseModel):
    sfen: str
    candidates: list[CandidateMove] = []
    move_count: int = 0
    level: str = "intermediate"
    backend: Optional[str] = None   # "ollama" | "claude"（省略時は環境変数）
    model: Optional[str] = None     # モデル名（省略時はデフォルト）


class ExplainResponse(BaseModel):
    commentary: str
    backend_used: str
    model_used: str


@router.post("/explain", response_model=ExplainResponse)
async def explain_position(req: ExplainRequest) -> ExplainResponse:
    try:
        shogi.Board(req.sfen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SFEN: {e}")

    if req.level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"level は {', '.join(_VALID_LEVELS)} のいずれかを指定してください",
        )

    backend_name = req.backend or _DEFAULT_BACKEND
    if backend_name not in _VALID_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"backend は {', '.join(_VALID_BACKENDS)} のいずれかを指定してください",
        )

    candidates_dict = [c.model_dump() for c in req.candidates]

    try:
        ctx = build_context(req.sfen, candidates_dict, req.move_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"盤面解析エラー: {e}")

    backend = make_backend(backend_name, req.model)
    commentator = Commentator(backend)

    try:
        commentary = await commentator.explain(ctx, level=req.level)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLMエラー: {e}")

    return ExplainResponse(
        commentary=commentary,
        backend_used=backend.name,
        model_used=backend.model,
    )
