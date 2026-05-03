from dataclasses import dataclass


@dataclass
class RuleSet:
    """ルールセット。標準将棋はすべてデフォルト値。"""
    allow_double_pawn: bool = False               # 二歩あり
    allow_pawn_on_last_rank: bool = False          # 端歩なし（1段目の歩打ちを許可）
    king_moves_only_when_in_check: bool = False    # 取られるときしか玉を動かせない
    allow_arbitrary_start: bool = False            # 任意の初期配置（変則盤面）を許可
