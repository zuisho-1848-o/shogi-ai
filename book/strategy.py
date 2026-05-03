from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    """戦法定義。定跡ファイル内のタグでフィルタリングする。

    tag が None の場合はすべての定跡手を対象とする。
    tag を指定した場合は同一タグを持つ定跡手のみを優先する。

    Examples:
        Strategy(name="ranging_rook", tag="ranging_rook")
        Strategy(name="free")   # tag=None ですべての手を許可
    """

    name: str
    tag: str | None = None
    # 序盤に優先的に指したい USI 手のリスト（定跡外の局面でも参照）
    preferred_moves: list[str] = field(default_factory=list)


# プリセット戦法
STRATEGY_FREE = Strategy(name="free")
STRATEGY_RANGING_ROOK = Strategy(name="ranging_rook", tag="ranging_rook")
STRATEGY_STATIC_ROOK = Strategy(name="static_rook", tag="static_rook")

STRATEGY_MAP: dict[str, Strategy] = {
    "free": STRATEGY_FREE,
    "ranging_rook": STRATEGY_RANGING_ROOK,
    "static_rook": STRATEGY_STATIC_ROOK,
}
