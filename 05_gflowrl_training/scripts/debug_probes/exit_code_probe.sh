# Minimal reproduction: the wrap-up structure of run_rl.sh
#   set -e + trap cleanup EXIT + cleanup kills a background job and then waits on it
# Question: after training exits successfully, what is the exit code of the whole script?
set -euxo pipefail
cleanup(){
  echo CLEANUP_START
  [ -n "$P" ] && kill $P 2>/dev/null && wait $P 2>/dev/null
  echo CLEANUP_END
}
trap cleanup EXIT
sleep 300 &
P=$!
echo TRAINING_COMPLETE
