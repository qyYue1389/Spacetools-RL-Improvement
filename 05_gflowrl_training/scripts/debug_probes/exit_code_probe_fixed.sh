# The same reproduction, with set +e added at the start of cleanup — exactly the line patched into run_rl.sh
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
