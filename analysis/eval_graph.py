"""評価値推移グラフを matplotlib で PNG として生成する。"""
from __future__ import annotations

from pathlib import Path


def save_eval_graph(
    scores: list[int],
    output_path: Path,
    *,
    title: str = "評価値推移",
) -> None:
    """評価値（centipawn）の推移グラフを PNG として保存する。

    scores[i] は i+1 手目終了時点の「先手視点」評価値。
    正 = 先手優勢、負 = 後手優勢。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib が必要です: pip install 'shogi-ai[full]'"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if scores:
        moves = list(range(1, len(scores) + 1))
        ax.fill_between(
            moves,
            [max(0, s) for s in scores],
            alpha=0.35,
            color="#3060c0",
            label="先手優勢",
        )
        ax.fill_between(
            moves,
            [min(0, s) for s in scores],
            alpha=0.35,
            color="#c03030",
            label="後手優勢",
        )
        ax.plot(moves, scores, color="#80c8ff", linewidth=1.5)

    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
    ax.set_xlabel("手数", color="#ccc")
    ax.set_ylabel("評価値 (cp)", color="#ccc")
    ax.set_title(title, color="#f0c040", fontsize=11)
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if scores:
        ax.legend(
            facecolor="#16213e",
            edgecolor="#444",
            labelcolor="#ccc",
            fontsize=8,
            loc="upper right",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
