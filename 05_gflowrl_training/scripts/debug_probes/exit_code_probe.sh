# 最小复现:run_rl.sh 的收尾结构
#   set -e + trap cleanup EXIT + cleanup 里 kill 一个后台 job 再 wait 它
# 问题:训练成功退出之后,整个脚本的退出码是多少?
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
