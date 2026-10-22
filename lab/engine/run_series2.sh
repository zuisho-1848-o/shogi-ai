#!/bin/zsh
# エンジン対局の第2弾: 枝刈りを足した nonnn2（1手0.2秒、各60局、省電力コアで1つずつ）
cd "$(dirname "$0")"
for pair in "nonnn2:0.5 nonnn:0.5" "nonnn2:0.5 nonnn2:0" "nonnn2:0.5 material"; do
  set -- ${=pair}
  taskpolicy -b nice -n 19 ../.venv/bin/python match.py --a $1 --b $2 --a-ms 200 --b-ms 200 --n 30
done
