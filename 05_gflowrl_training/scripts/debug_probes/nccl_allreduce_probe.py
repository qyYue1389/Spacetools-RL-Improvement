import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
from datetime import timedelta

def w(rank, world):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = os.environ.get('PROBE_PORT', '29511')
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=40))
    t = torch.ones(1024, 1024, device='cuda')
    dist.all_reduce(t)
    torch.cuda.synchronize()
    if rank == 0:
        print('ALLREDUCE_OK sum=%.0f expect=%.0f' % (t.sum().item(), 1024*1024*world), flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mp.spawn(w, args=(n,), nprocs=n)
