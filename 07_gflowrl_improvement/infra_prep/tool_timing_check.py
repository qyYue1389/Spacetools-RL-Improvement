"""
Self-check for the per-call tool timing (patched/tool_agent_loop.py + Toolshed's
toolshed/integration/verl.py with toolshed_tool_timing_vs_712e557.diff applied).

Extracts ToolAgentLoop._call_tool and ToolshedMethodTool.execute from the patched sources (and
_call_tool from the unpatched SpaceTools-RL for comparison), runs them on stub tools, and checks:
  1. with TOOL_TIMING_DIR unset: no file written, return values identical to the original
  2. with it set: one record per call; latency ~= exec_wait + remote + post-processing;
     exec_wait grows when calls exceed the executor's threads (4 here); errors -> ok False;
     unknown tool -> ok False, exec_wait None; return values still identical
usage: python tool_timing_check.py PATCHED_TOOL_AGENT_LOOP ORIG_TOOL_AGENT_LOOP PATCHED_TOOLSHED_VERL
"""
import ast, asyncio, glob, json, logging, os, sys, tempfile, textwrap, time, types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PATCHED, ORIG, TSV = sys.argv[1:4]
fails = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)

def method_src(path, cls, name):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == name:
                    seg = ast.get_source_segment(src, f)
                    deco = "".join(" " * f.col_offset + "@" + ast.get_source_segment(src, d) + "\n" for d in f.decorator_list)
                    return textwrap.dedent(deco + " " * f.col_offset + seg)
    raise SystemExit(f"{cls}.{name} not in {path}")

def func_src(path, name):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"{name} not in {path}")

class ToolResponse:
    def __init__(self, text=None, image=None, video=None):
        self.text, self.image, self.video = text, image, video
    def key(self):
        return (self.text, self.image, self.video)

class ToolResult:
    def __init__(self, text, is_error=False):
        self.text, self.is_error = text, is_error
    def to_dict(self):
        return {"text": self.text}

sys.modules["ray"] = types.SimpleNamespace(is_initialized=lambda: False)
log = logging.getLogger("check")
base = {"json": json, "os": os, "time": time, "asyncio": asyncio, "Counter": Counter, "logger": log,
        "ToolResponse": ToolResponse, "ToolResult": ToolResult, "Any": object, "Tuple": tuple,
        "FunctionCall": object, "AgentData": object, "_resolve_variables": lambda a, c: a,
        "rollout_trace_op": lambda f: f, "_ensure_weave_initialized": lambda cfg: None}

def load_agent(path, timing_dir):
    ns = dict(base, _TOOL_TIMING_DIR=timing_dir, _TOOL_INFLIGHT=Counter())
    if timing_dir:
        exec(func_src(path, "_record_tool_call"), ns)
    exec(method_src(path, "ToolAgentLoop", "_call_tool"), ns)
    return ns["_call_tool"]

ns_t = dict(base)
exec(method_src(TSV, "ToolshedMethodTool", "execute"), ns_t)
execute = ns_t["execute"]

class FakeTool:                          # ToolshedMethodTool with a fake remote function
    def __init__(self, name, delay, fail=False):
        self.name, self.delay, self.fail = name, delay, fail
        self._instance_dict, self.config = {}, {"function_name": name + ".m"}
    def _get_function_wrapper(self):
        def f(**kw):
            time.sleep(self.delay)
            return ToolResult(f"{self.name}:{kw.get('x')}", is_error=self.fail)
        return f
    async def create(self, create_kwargs=None):
        iid = str(len(self._instance_dict)) + self.name + str(time.time())
        self._instance_dict[iid] = {"calls": 0, "last_result": None, "total_reward": 0.0}
        return iid, ToolResponse()
    async def release(self, iid):
        self._instance_dict.pop(iid, None)
    execute = execute

class Call:
    def __init__(self, name, x):
        self.name, self.arguments = name, json.dumps({"x": x})

class Loop:
    max_tool_response_length, tool_response_truncate_side = 10_000, "middle"
    def __init__(self):
        self.tools = {"slow": FakeTool("slow", 0.05), "fast": FakeTool("fast", 0.0),
                      "bad": FakeTool("bad", 0.0, fail=True)}

AD = types.SimpleNamespace(context_vars={}, image_data=[])
CALLS = [Call("slow", i) for i in range(12)] + [Call("fast", i) for i in range(4)] + [Call("bad", 0), Call("nope", 0)]

async def run(fn):
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=4))
    lp = Loop()
    out = await asyncio.gather(*[fn(lp, c, {}, AD) for c in CALLS])
    return [(r.key(), rw, sorted(m.keys()) if isinstance(m, dict) else m) for r, rw, m in out]

ref = asyncio.run(run(load_agent(ORIG, None)))

print("1) TOOL_TIMING_DIR unset")
d0 = tempfile.mkdtemp()
out0 = asyncio.run(run(load_agent(PATCHED, None)))
check("return values identical to original", out0 == ref)
check("no file written", not os.listdir(d0))

print("2) TOOL_TIMING_DIR set")
d1 = tempfile.mkdtemp()
out1 = asyncio.run(run(load_agent(PATCHED, d1)))
check("return values identical to original", out1 == ref)
recs = [json.loads(l) for f in glob.glob(d1 + "/*.jsonl") for l in open(f)]
check("one record per call", len(recs) == len(CALLS), f"({len(recs)})")
by = lambda n: [r for r in recs if r["tool"] == n]
slow = by("slow")
check("slow: remote ~= 0.05 s", all(0.045 < r["remote"] < 0.2 for r in slow),
      f"(median {sorted(r['remote'] for r in slow)[len(slow)//2]:.3f})")
check("slow: latency >= exec_wait + remote", all(r["latency"] + 1e-6 >= r["exec_wait"] + r["remote"] for r in slow))
waits = sorted(r["exec_wait"] for r in slow)
check("exec_wait grows when calls > executor threads", waits[-1] > 0.08, f"(max {waits[-1]:.3f} s, executor {slow[0]['executor_workers']})")
check("in-flight counted", max(r["inflight_all"] for r in recs) >= 12)
check("toolshed error -> ok False", [r["ok"] for r in by("bad")] == [False])
check("unknown tool -> ok False, exec_wait None", by("nope")[0]["ok"] is False and by("nope")[0]["exec_wait"] is None)
check("successful calls ok True", all(r["ok"] for r in slow + by("fast")))

print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
