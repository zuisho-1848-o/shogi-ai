#!/bin/zsh
# エンジン対局の第3弾: 脅威の特徴量を外した軽い線形評価（1手0.2秒、各60局、省電力コアで1つずつ）
cd "$(dirname "$0")"
for pair in "nonnn2:0.5:kp nonnn2:0" "nonnn2:0.5:kp material"; do
  set -- ${=pair}
  taskpolicy -b nice -n 19 ../.venv/bin/python match.py --a $1 --b $2 --a-ms 200 --b-ms 200 --n 30
done
