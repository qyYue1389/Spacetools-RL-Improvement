#!/bin/bash
# SpaceTools SFT environment restore — target machine needs x86_64 Linux + A6000/A4000/A100 (sm_80 or sm_86)
#   sudo bash RESTORE.sh
set -euo pipefail

# ⚠️ Resolve at the top, before any cd — otherwise relative invocations resolve to the wrong directory
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$SELF_DIR/spacetools-sft-env.tar.zst"

echo "=== 0. Precondition checks (stop if any fails, do not go further) ==="

[ "$(id -u)" -eq 0 ] || { echo "✗ needs root (writes to /opt)"; exit 1; }
echo "  ✓ root"

[ "$(uname -m)" = "x86_64" ] || { echo "✗ only x86_64 is supported, current is $(uname -m)"; exit 1; }
echo "  ✓ x86_64"

# glibc: the self-built flash_attn.so in the package needs >= 2.32
# ⚠️ Do not use `| head -1`: under set -o pipefail, head closes the pipe after reading the first line,
# the upstream process gets SIGPIPE and exits 141, pipefail propagates it, and set -e aborts right away.
# This is a race — if upstream writes fast it passes by luck, if slow it fails at random. awk reads to EOF and does not trigger it.
GLIBC=$(ldd --version | awk 'NR==1{print $NF}')
if [ "$(printf '%s\n2.32\n' "$GLIBC" | sort -V | awk 'NR==1')" != "2.32" ]; then
    echo "✗ glibc $GLIBC < 2.32, flash_attn.so cannot be loaded"; exit 1
fi
echo "  ✓ glibc $GLIBC (>= 2.32)"

command -v nvidia-smi >/dev/null || { echo "✗ no nvidia-smi, the target machine needs the NVIDIA driver"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1{print $1}')
DRV_MAJ=${DRV%%.*}
[ "$DRV_MAJ" -ge 525 ] || { echo "✗ driver $DRV < 525, cannot run the CUDA 12.x runtime"; exit 1; }
echo "  ✓ driver $DRV (>= 525; no need to install the CUDA toolkit, the runtime libraries are in the package)"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1{print $1}')
case "$CC" in
  8.0|8.6) echo "  ✓ compute_cap $CC" ;;
  *) echo "  ✗ compute_cap $CC — the flash-attn in this package has only an sm_80 cubin and no PTX fallback."
     echo "    A6000/A4000/A100 (8.0/8.6) work; H100 (9.0)/Blackwell need flash-attn rebuilt."
     echo "    To continue anyway, set ALLOW_ARCH_MISMATCH=1"
     [ "${ALLOW_ARCH_MISMATCH:-0}" = "1" ] || exit 1 ;;
esac

AVAIL=$(df -BG --output=avail /opt 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -ge 12 ] || { echo "✗ only ${AVAIL}G left on the disk holding /opt, need >= 12G"; exit 1; }
echo "  ✓ /opt free space ${AVAIL}G"

command -v zstd >/dev/null || { echo "✗ needs zstd (apt install zstd)"; exit 1; }
[ -f "$ARCHIVE" ] || { echo "✗ cannot find $ARCHIVE"; exit 1; }

echo
echo "=== 1. Verify archive integrity ==="
if [ -f "$SELF_DIR/SHA256SUMS" ]; then
    ( cd "$SELF_DIR" && sha256sum -c SHA256SUMS ) || { echo "✗ checksum mismatch, the package was corrupted in transfer"; exit 1; }
else
    echo "  ⚠️ no SHA256SUMS, skipping verification"
fi

echo
echo "=== 2. Paths that already exist ==="
for d in /opt/conda-st /workspace/SpaceTools-SFT /workspace/SpaceTools; do
    [ -e "$d" ] && echo "  ⚠️ $d already exists, unpacking will overwrite files with the same name" || true
done

echo
echo "=== 3. Unpack (about 9.5 GB, paths must stay as they are) ==="
mkdir -p /workspace
zstd -dc "$ARCHIVE" | tar -C / -xf -
echo "  ✓ unpack done"

echo
echo "=== 4. Acceptance check ==="
bash "$SELF_DIR/VERIFY.sh"
rc=$?

echo
if [ "$rc" -eq 0 ]; then
cat <<'TXT'
==========================================
Environment ready. In every new shell:
  export PATH=/opt/conda-st/bin:$PATH && eval "$(conda shell.bash hook)"
  conda activate spacetools-sft
  export HF_HOME=/workspace/hf TMPDIR=/workspace/tmp
==========================================
TXT
fi
exit "$rc"
