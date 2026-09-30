# 同一个复现,cleanup 开头加了 set +e —— 就是打进 run_rl.sh 的那一行
set -euxo pipefail
cleanup(){
  set +e
  echo CLEANUP_START
  [ -n "$P" ] && kill $P 2>/dev/null && wait $P 2>/dev/null
  echo RAY_STOP_WOULD_RUN_HERE
  echo CLEANUP_END
}
trap cleanup EXIT
sleep 300 &
P=$!
echo TRAINING_COMPLETE
