#!/usr/bin/env python3
"""All-reduce correctness + bandwidth probe for the training GPUs (runs in ~10-40 s).

    python3 nccl_bw_probe.py 4                      # 4 GPUs, NCCL
    python3 nccl_bw_probe.py 4 --backend gloo       # CPU dry run of the script itself

Prints one line:  PROBE_OK n=4 size=256MB busbw=XX.X GB/s   (or times out / errors).
busbw uses the nccl-tests convention for all-reduce: algbw * 2 * (n-1) / n.
Extends training/GFlowRL/A40/scripts/ncclprobe.py (correctness only) with a bandwidth number,
so a machine with working-but-slow P2P is caught too, not only a hanging one.
"""
import argparse, os, time
from datetime import timedelta
import torch, torch.distributed as dist, torch.multiprocessing as mp


def work(rank, n, a):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(a.port), RANK=str(rank), WORLD_SIZE=str(n))
    dev = torch.device("cpu")
    if a.backend == "nccl":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    dist.init_process_group(a.backend, timeout=timedelta(seconds=a.timeout))
    numel = a.mb * 2**20 // 4
    t = torch.ones(numel, device=dev)
    dist.all_reduce(t)                                   # warm-up + correctness
    ok = bool((t == n).all())
    sync = torch.cuda.synchronize if a.backend == "nccl" else (lambda: None)
    sync(); dist.barrier()
    t0 = time.time()
    for _ in range(a.iters):
        dist.all_reduce(t)
    sync()
    dt = (time.time() - t0) / a.iters
    if rank == 0:
        algbw = numel * 4 / dt / 1e9
        busbw = algbw * 2 * (n - 1) / n
        print(f"{'PROBE_OK' if ok else 'PROBE_WRONG_RESULT'} n={n} size={a.mb}MB "
              f"time={dt * 1e3:.1f}ms busbw={busbw:.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=4)
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=40)
    ap.add_argument("--port", type=int, default=29500 + os.getpid() % 400)
    a = ap.parse_args()
    mp.spawn(work, args=(a.n, a), nprocs=a.n)
