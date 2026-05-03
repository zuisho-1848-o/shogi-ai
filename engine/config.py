from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from core.rules import RuleSet


@dataclass
class EngineConfig:
    search: Literal["minimax", "alphabeta", "mcts"] = "alphabeta"
    eval: Literal["material", "pst", "kpp", "nnue"] = "material"
    board_impl: Literal["python_shogi", "native"] = "python_shogi"
    depth: int = 5
    time_limit_ms: int = 3000
    multi_pv: int = 5
    nnue_model_path: Path = field(default_factory=lambda: Path("models/nnue.onnx"))
    kpp_table_path: Path = field(default_factory=lambda: Path("models/kpp.bin"))
    opening_book_path: Path | None = field(default_factory=lambda: Path("book/standard.sfen"))
    strategy: str | None = None
    rules: RuleSet = field(default_factory=RuleSet)
