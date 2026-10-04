"""
Infra 自检:+trainer.dump_images=false 时 rollout / eval dump 只写 JSONL,不存 PNG;不设时行为不变。

与其它自检同样的纪律:_dump_generations 从打了补丁的 ray_trainer.py 抽源码 exec,不另抄一份。
用法: python dump_images_check.py PATCHED_RAY_TRAINER
"""
import ast, json, os, sys, tempfile, textwrap, types
import numpy as np

src = open(sys.argv[1]).read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_dump_generations")
ns = {"os": os, "json": json, "np": np}
exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)


class Img:
    def save(self, path):
        open(path, "wb").write(b"png")


def run(trainer_cfg):
    d = tempfile.mkdtemp()
    me = types.SimpleNamespace(global_steps=7, config=types.SimpleNamespace(trainer=trainer_cfg))
    ns["_dump_generations"](me, ["q1", "q2"], ["a1", "a2"], [None, None], [1.0, 0.0], {}, d,
                            multi_modal_data=[{"image": [Img(), Img()]}, {"image": Img()}])
    rows = [json.loads(l) for l in open(os.path.join(d, "7.jsonl"))]
    imgs = os.path.join(d, "images_7")
    return rows, (sorted(os.listdir(imgs)) if os.path.isdir(imgs) else None)


rows, imgs = run({})                                   # key absent -> old behavior
assert len(rows) == 2 and imgs == ["s000_img0000.png", "s000_img0001.png", "s001_img0000.png"], imgs
rows_on, imgs_on = run({"dump_images": True})
assert rows_on == rows and imgs_on == imgs
rows_off, imgs_off = run({"dump_images": False})
assert rows_off == rows and imgs_off is None, imgs_off  # same JSONL, no images dir at all
print("RESULT: PASS")
