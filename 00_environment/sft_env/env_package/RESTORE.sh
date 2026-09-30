#!/bin/bash
# SpaceTools SFT 环境恢复 —— 目标机需 x86_64 Linux + A6000/A4000/A100(sm_80 或 sm_86)
#   sudo bash RESTORE.sh
set -euo pipefail

# ⚠️ 顶部解析,在任何 cd 之前 —— 否则相对调用会解析到错的目录
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$SELF_DIR/spacetools-sft-env.tar.zst"

echo "=== 0. 前置检查(任何一项不过就停,不往下走)==="

[ "$(id -u)" -eq 0 ] || { echo "✗ 需要 root(要写 /opt)"; exit 1; }
echo "  ✓ root"

[ "$(uname -m)" = "x86_64" ] || { echo "✗ 只支持 x86_64,当前 $(uname -m)"; exit 1; }
echo "  ✓ x86_64"

# glibc:包里自编的 flash_attn.so 需要 >= 2.32
# ⚠️ 不要用 `| head -1`:set -o pipefail 下,head 读完第一行就关管道,
# 上游进程收到 SIGPIPE 退出 141,pipefail 把它传出来,set -e 直接中止。
# 这是竞态 —— 上游写得快就侥幸通过,慢就随机失败。awk 读到 EOF,不会触发。
GLIBC=$(ldd --version | awk 'NR==1{print $NF}')
if [ "$(printf '%s\n2.32\n' "$GLIBC" | sort -V | awk 'NR==1')" != "2.32" ]; then
    echo "✗ glibc $GLIBC < 2.32,flash_attn.so 加载不了"; exit 1
fi
echo "  ✓ glibc $GLIBC (>= 2.32)"

command -v nvidia-smi >/dev/null || { echo "✗ 没有 nvidia-smi,目标机需要 NVIDIA 驱动"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1{print $1}')
DRV_MAJ=${DRV%%.*}
[ "$DRV_MAJ" -ge 525 ] || { echo "✗ 驱动 $DRV < 525,跑不了 CUDA 12.x 运行时"; exit 1; }
echo "  ✓ 驱动 $DRV (>= 525;不需要装 CUDA toolkit,运行时库在包里)"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1{print $1}')
case "$CC" in
  8.0|8.6) echo "  ✓ compute_cap $CC" ;;
  *) echo "  ✗ compute_cap $CC —— 本包的 flash-attn 只有 sm_80 cubin 且无 PTX 回退。"
     echo "    A6000/A4000/A100(8.0/8.6)可用;H100(9.0)/Blackwell 需要重编 flash-attn。"
     echo "    确认要继续就设 ALLOW_ARCH_MISMATCH=1"
     [ "${ALLOW_ARCH_MISMATCH:-0}" = "1" ] || exit 1 ;;
esac

AVAIL=$(df -BG --output=avail /opt 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -ge 12 ] || { echo "✗ /opt 所在盘只剩 ${AVAIL}G,需要 >= 12G"; exit 1; }
echo "  ✓ /opt 可用空间 ${AVAIL}G"

command -v zstd >/dev/null || { echo "✗ 需要 zstd(apt install zstd)"; exit 1; }
[ -f "$ARCHIVE" ] || { echo "✗ 找不到 $ARCHIVE"; exit 1; }

echo
echo "=== 1. 校验档案完整性 ==="
if [ -f "$SELF_DIR/SHA256SUMS" ]; then
    ( cd "$SELF_DIR" && sha256sum -c SHA256SUMS ) || { echo "✗ 校验和不匹配,包在传输中损坏了"; exit 1; }
else
    echo "  ⚠️ 没有 SHA256SUMS,跳过校验"
fi

echo
echo "=== 2. 已存在的路径 ==="
for d in /opt/conda-st /workspace/SpaceTools-SFT /workspace/SpaceTools; do
    [ -e "$d" ] && echo "  ⚠️ $d 已存在,解包会覆盖同名文件" || true
done

echo
echo "=== 3. 解包(约 9.5 GB,路径必须保持原样)==="
mkdir -p /workspace
zstd -dc "$ARCHIVE" | tar -C / -xf -
echo "  ✓ 解包完成"

echo
echo "=== 4. 验收 ==="
bash "$SELF_DIR/VERIFY.sh"
rc=$?

echo
if [ "$rc" -eq 0 ]; then
cat <<'TXT'
==========================================
环境就绪。每次新 shell:
  export PATH=/opt/conda-st/bin:$PATH && eval "$(conda shell.bash hook)"
  conda activate spacetools-sft
  export HF_HOME=/workspace/hf TMPDIR=/workspace/tmp
==========================================
TXT
fi
exit "$rc"
