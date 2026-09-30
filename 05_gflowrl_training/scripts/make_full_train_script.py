s = open('/root/smoke_train.sh', encoding='utf-8').read()
for a, b in (("/workspace/exp/p7_cprime_smoke", "/workspace/exp/p7_cprime"),
             ("3 步冒烟", "全量 86 步(单 seed)"),
             ("SAVE_FREQ=-1", "SAVE_FREQ=5"),
             ("    trainer.total_training_steps=3 \\\n", ""),
             ("smoke_train.log", "full_train.log"),
             ("status_smoke", "status_full"),
             ("SMOKE_", "FULL_")):
    assert a in s, a[:40]
    s = s.replace(a, b)
s = "".join(x for x in s.splitlines(True) if "sfn" not in x)
open('/root/full_train.sh', 'w', encoding='utf-8').write(s)
print("wrote /root/full_train.sh")
