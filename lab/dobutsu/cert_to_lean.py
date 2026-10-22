"""証明書（cert.bin）を Lean のソースに埋め込む（Lean の証明はファイルを読めないので、データをソースに入れる）

base64 の文字列を 1MB ずつの定義に分けて書き出す。復号と読み込みは lab/lean/Dobutsu/Main.lean が行う。
出力: ../lean/Dobutsu/Data.lean（git 管理外。大きいので）
使い方: python cert_to_lean.py
"""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../data/dobutsu/cert.bin")
OUT = os.path.join(HERE, "../lean/Dobutsu/Data.lean")
CHUNK = 1 << 20


def main():
    import sys

    raw = open(sys.argv[1] if len(sys.argv) > 1 else SRC, "rb").read()
    b64 = base64.b64encode(raw).decode()
    chunks = [b64[i : i + CHUNK] for i in range(0, len(b64), CHUNK)]
    with open(OUT, "w") as f:
        f.write("-- 自動生成（lab/dobutsu/cert_to_lean.py）。どうぶつしょうぎ「後手必勝」の証明書（base64）\n")
        f.write("namespace Dobutsu.Data\n\n")
        for i, c in enumerate(chunks):
            f.write(f'def chunk{i} : String := "{c}"\n')
        f.write("\ndef all : String := String.join [" + ", ".join(f"chunk{i}" for i in range(len(chunks))) + "]\n")
        f.write("\nend Dobutsu.Data\n")
    print(f"{len(raw):,} bytes -> {len(b64):,} chars in {len(chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()
