#!/bin/zsh
# 裏で回している実験プロセスの見張り役。
# システムの空きメモリが LOW% を下回ったら対象プロセスを一時停止(SIGSTOP)し、HIGH% を超えたら再開(SIGCONT)する。
# 対象: コマンドラインに PATTERN を含むプロセス（既定: lab の solve / sample / analyze_positions / x1。ユーザーが使うやねうら王は止めない）
# 使い方: nohup lab/tools/guard.sh > lab/data/guard.log 2>&1 &
LOW=${LOW:-15}
HIGH=${HIGH:-25}
PATTERN=${PATTERN:-'^(\./solve|[^ ]*dobutsu/sample|[^ ]*python[0-9.]* [^ ]*(analyze_positions|x1_expressivity|x1b_errors|x2_search|x4_patterns|x5_threats|x7_expressivity))'}
paused=0
while true; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  pids=$(pgrep -f -- "$PATTERN" | tr '\n' ' ')
  if [[ -n "$pids" ]]; then
    if (( free < LOW && paused == 0 )); then
      kill -STOP ${=pids} 2>/dev/null; paused=1; echo "$(date '+%F %T') free=${free}% -> STOP $pids"
    elif (( free > HIGH && paused == 1 )); then
      kill -CONT ${=pids} 2>/dev/null; paused=0; echo "$(date '+%F %T') free=${free}% -> CONT $pids"
    fi
  else
    paused=0
  fi
  sleep 20
done
