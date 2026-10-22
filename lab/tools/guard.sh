#!/bin/zsh
# 裏で回している実験プロセスの見張り役。
# システムの空きメモリが LOW% を下回ったら対象プロセスを一時停止(SIGSTOP)し、HIGH% を超えたら再開(SIGCONT)する。
# 対象: コマンドラインに PATTERN を含むプロセス（既定: lab の solve / sample / analyze_positions / x1。ユーザーが使うやねうら王は止めない）
# 使い方: nohup lab/tools/guard.sh > lab/data/guard.log 2>&1 &
LOW=${LOW:-15}
HIGH=${HIGH:-25}
PATTERN=${PATTERN:-'^(\./solve|[^ ]*dobutsu/sample|[^ ]*python[0-9.]* [^ ]*(analyze_positions|x1_expressivity|x1b_errors|x2_search|x4_patterns|x5_threats|x7_expressivity|x7b_qsearch|x8_rules|x9_long_wins|x10_theory|x6_mate_zones|x11_match|x12_rules|x12b_verify|x13_mix|x14_rules|x15_proverbs9|x16_reversal_tree|x17_teacher|x18_lowrank|x19_lowrank9|x21_searchpos))'}
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
