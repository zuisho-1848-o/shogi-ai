#!/bin/zsh
# 非NN評価の研究用エンジンをビルドする。cshogi（GPL）のソースは third_party/cshogi に置く（git 管理外）
#   git clone --depth 1 https://github.com/TadaoYamaoka/cshogi.git third_party/cshogi
#   新しい clang では src/square.hpp の SQWW_R / SQWW_U に int() を付けるパッチが必要（research/運用メモ.md）
set -e
cd "$(dirname "$0")"
SRC=third_party/cshogi/src
mkdir -p build
clang++ -O3 -std=c++17 -mcpu=apple-m1 -DNDEBUG -I$SRC \
  engine.cpp $SRC/bitboard.cpp $SRC/common.cpp $SRC/generateMoves.cpp $SRC/hand.cpp $SRC/init.cpp \
  $SRC/move.cpp $SRC/mt64bit.cpp $SRC/position.cpp $SRC/square.cpp $SRC/usi.cpp $SRC/search.cpp $SRC/book.cpp \
  -o build/nonnn
echo built build/nonnn
