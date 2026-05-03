from __future__ import annotations

import sys

from engine.engine import Engine

ENGINE_NAME = "shogi-ai"
ENGINE_AUTHOR = "shogi-ai"


def run_usi_loop(engine: Engine) -> None:
    """USI プロトコルのメインループ。"""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "usi":
            print(f"id name {ENGINE_NAME}")
            print(f"id author {ENGINE_AUTHOR}")
            print("option name MultiPV type spin default 5 min 1 max 10")
            print("usiok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "usinewgame":
            engine.new_game()
        elif cmd == "setoption":
            # setoption name <Name> value <Value>
            if len(tokens) >= 4 and tokens[1] == "name":
                try:
                    val_idx = tokens.index("value", 2)
                    name = " ".join(tokens[2:val_idx])
                    value = " ".join(tokens[val_idx + 1 :])
                    engine.set_option(name, value)
                except ValueError:
                    pass
        elif cmd == "position":
            engine.set_position(tokens[1:])
        elif cmd == "go":
            engine.go(tokens[1:])
        elif cmd == "stop":
            engine.stop()
        elif cmd == "quit":
            break

        sys.stdout.flush()
