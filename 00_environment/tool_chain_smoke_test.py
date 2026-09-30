#!/usr/bin/env python3
"""P2 smoke test: exercise every v1 tool once, in dependency order.

This is the tool-level coverage step of P2. It deliberately uses NO language model:
it drives the Toolshed tools directly, so a failure here is unambiguously a tool
problem rather than an orchestration or parsing problem.

The call order below is the BOP-ASK chain, which is the longest one the eval uses:

    depth_estimator.estimate_depth_with_pointcloud(image)  -> point_cloud, focal_length_px
    roborefer.detect_one(image, obj)                       -> (x, y)
    vlm.detect_one(image, obj)                             -> (x, y)
    vision_ops.index_at(depth_map, x, y)                   -> depth at that pixel
    sam2.segment_from_point(image, x, y)                   -> mask
    bounding_box.compute_bbox(point_cloud, mask, focal)    -> oriented 3D box
    grasp_generator.compute_grasp(pc, mask, image, focal)  -> grasp poses

TOOL_CONFIGS mirrors examples/toolshed/run_eval.sh (v1 branch) so that what this
script exercises is exactly what the eval will run. KEEP THE TWO IN SYNC: this
copy was left on the superseded interim fractions (vlm/roborefer 0.55, sam2 and
depth_estimator 0.15) after run_eval.sh moved to the final split in 910f6f5d.
On 48 GB cards that mistake was survivable -- Molmo 33 + DepthPro 12.5 + SAM2 1.2
just fits 48 -- so it went unnoticed through all of P2. On the 40 GB A100s it
OOMs on the first depth call, because 0.55 + 0.15 lets Ray pack Molmo, a
DepthPro actor and a SAM2 actor onto one card.

Prerequisites:
    ray start --head --num-gpus=4 --port=6379
    export ROBOREFER_MODEL=/workspace/models/RoboRefer-8B-SFT
    export DEPTH_CHECKPOINT=/workspace/models/depth_pro.pt

Usage:
    python p2_tool_chain.py [--image PATH] [--object NAME]
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

import ray
from toolshed import start_toolkit, get_toolkit, shutdown_toolkit


# --------------------------------------------------------------------------
# Reporting helpers. Every check prints PASS/FAIL/WARN plus the observed value,
# because the observed value is the point: these tools fail silently by
# returning plausible-looking numbers in the wrong unit or coordinate frame.
# --------------------------------------------------------------------------
RESULTS = []


def check(name, ok, detail):
    tag = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    RESULTS.append((tag, name))
    print(f"  [{tag}] {name}: {detail}")


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


class Step:
    """Isolate one tool step.

    Each tool gets its own try/except so a failure in one does not hide the
    behaviour of the others: a single run costs minutes of model loading, so it
    should teach us about every part of the chain, not just the first break.
    """

    def __init__(self, title):
        self.title = title

    def __enter__(self):
        section(self.title)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            check(self.title, False, f"{exc_type.__name__}: {str(exc)[:250]}")
            traceback.print_exc()
            return True  # suppressed: continue with the remaining steps
        return False


def build_tool_configs(roborefer_model, depth_checkpoint):
    """Verbatim copy of the v1 TOOL_CONFIGS block in run_eval.sh."""
    return {
        'roborefer': {'num_actors': 1, 'resources': {'num_gpus': 0.6},
                      'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
                      'args': {'model_path': roborefer_model, 'no_output_image': True,
                               'no_output_vars': False, 'exclude_methods': ['general_query'],
                               'exclude_behavior': 'error'}},
        'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0},
                'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
                'args': {'no_output_image': True, 'no_output_vars': False,
                         'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'auto',
                         'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
        'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2},
                 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
                 'args': {'no_output_image': True, 'no_output_vars': False}},
        'depth_estimator': {'num_actors': 2, 'resources': {'num_gpus': 0.4},
                            'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
                            'args': {'checkpoint_path': depth_checkpoint,
                                     'no_output_image': True, 'no_output_vars': False}},
        'bounding_box': {'num_actors': 2, 'resources': {'num_gpus': 0.05},
                         'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
                         'args': {'no_output_image': True, 'no_output_vars': False}},
        'vision_ops': {'num_actors': 1, 'resources': {'num_gpus': 0},
                       'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
                       'args': {'no_output_image': True, 'no_output_vars': False,
                                'exclude_methods': ['mask_crop', 'point_crop',
                                                    'project_2d_points_to_3d',
                                                    'project_3d_points_to_2d', 'draw_polygon'],
                                'exclude_behavior': 'error'}},
        'grasp_generator': {'num_actors': 1, 'resources': {'num_gpus': 0.1},
                            'conda_env': 'spacetools-tool-graspgen', 'timeout': 600,
                            'args': {'no_output_image': True, 'no_output_vars': False}},
    }


def as_array(x):
    """Tool variables may arrive as list, tuple or ndarray."""
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def report_point(tool_name, result, width, height):
    """Report a detection point and, crucially, which coordinate frame it is in.

    A point returned normalized (0..1) but consumed as pixels -- or the reverse --
    is the single most damaging silent error in this pipeline, because every
    downstream geometric step still succeeds and simply returns the wrong answer.

    Measured convention: the tools speak NORMALIZED coordinates end to end.
    roborefer returns e.g. [(0.344, 0.674)]; vision_ops.index_at multiplies by the
    array dimensions; sam2.segment_from_point rejects anything outside [0, 1].
    This function therefore always returns normalized coordinates.
    """
    val = result.value
    print(f"    raw value: {val!r}")
    print(f"    text     : {result.text[:200] if result.text else None}")

    pt = None
    if isinstance(val, dict):
        for key in ("point", "points", "coordinates", "coords"):
            if key in val:
                pt = val[key]
                break
    else:
        pt = val
    arr = as_array(pt).astype(float).reshape(-1)
    if arr.size < 2:
        check(f"{tool_name} point shape", False, f"expected >=2 numbers, got {arr!r}")
        return None

    x, y = float(arr[0]), float(arr[1])
    normalized = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    in_pixels = 1.0 < x <= width and 1.0 < y <= height
    if normalized:
        check(f"{tool_name} coordinate frame", True,
              f"NORMALIZED (x={x:.4f}, y={y:.4f}) -> pixel ({x * width:.1f}, {y * height:.1f})")
        return x, y
    if in_pixels:
        check(f"{tool_name} coordinate frame", None,
              f"PIXELS (x={x:.1f}, y={y:.1f}) on {width}x{height} -- normalizing for downstream")
        return x / width, y / height
    check(f"{tool_name} coordinate frame", False,
          f"x={x}, y={y} fits neither 0..1 nor the {width}x{height} image bounds")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",
                    default="/workspace/SpaceTools/SpaceTools-Toolshed/examples/media/kitchen.png")
    ap.add_argument("--object", default="the sink")
    args = ap.parse_args()

    roborefer_model = os.environ.get("ROBOREFER_MODEL")
    depth_checkpoint = os.environ.get("DEPTH_CHECKPOINT")
    if not roborefer_model or not depth_checkpoint:
        sys.exit("Set ROBOREFER_MODEL and DEPTH_CHECKPOINT first (source /workspace/env.sh).")

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    print(f"Image : {args.image} ({width}x{height})")
    print(f"Object: {args.object!r}")

    ray.init(address="auto")
    print("\nStarting toolkit (models load lazily on first call; expect a slow first hit)...")
    # Keep the returned handle alive for the whole run. start_toolkit is called
    # with detached=False, so the router actor's lifetime is bound to this
    # handle: dropping it lets Ray garbage-collect the router out from under us,
    # and every later tool call fails with "Could not find ToolRouterActor".
    router = start_toolkit(build_tool_configs(roborefer_model, depth_checkpoint),
                           detached=False, dashboard=False)
    assert router is not None
    toolkit = get_toolkit()

    depth_map = point_cloud = focal = None
    px = py = None
    mask = None
    pt = pt_vlm = None

    # ------------------------------------------------------------------
    # 1. Depth. Everything geometric downstream depends on this being
    #    metric and on focal_length_px being sane for the image width.
    # ------------------------------------------------------------------
    with Step("1/7  depth_estimator.estimate_depth_with_pointcloud"):
        t0 = time.time()
        res = toolkit.depth_estimator.estimate_depth_with_pointcloud(image)
        print(f"    elapsed: {time.time() - t0:.1f}s")
        print(f"    text   : {res.text[:300] if res.text else None}")
        v = res.variables or {}
        print(f"    variables: {sorted(v.keys())}")

        depth_map = as_array(v["depth_map"]) if "depth_map" in v else None
        point_cloud = as_array(v["point_cloud"]) if "point_cloud" in v else None
        focal = float(v["focal_length_px"]) if "focal_length_px" in v else None

        if depth_map is not None:
            lo, hi, med = float(depth_map.min()), float(depth_map.max()), float(np.median(depth_map))
            check("depth map shape", depth_map.shape[:2] == (height, width),
                  f"{depth_map.shape} vs image (h={height}, w={width})")
            # Depth-Pro emits metres. Anything outside a room-scale range means the
            # unit is wrong (millimetres, or inverse depth), which flips every
            # relative-depth comparison downstream.
            check("depth in metres", 0.05 < med < 100.0,
                  f"min={lo:.3f} median={med:.3f} max={hi:.3f}")
        else:
            check("depth_map variable", False, "missing")

        if point_cloud is not None:
            check("point cloud shape", point_cloud.ndim == 2 and point_cloud.shape[1] == 3,
                  f"{point_cloud.shape}")
        else:
            check("point_cloud variable", False, "missing")

        if focal is not None:
            # A pinhole focal length for a normal lens lands near the image width.
            check("focal length plausible", 0.3 * width < focal < 4.0 * width,
                  f"{focal:.1f} px on a {width}px-wide image")
        else:
            check("focal_length_px variable", False, "missing")

        # ------------------------------------------------------------------
        # 2-3. The two pointing tools. The paper calls these point1 / point2.
        # ------------------------------------------------------------------
    with Step("2/7  roborefer.detect_one"):
        t0 = time.time()
        res = toolkit.roborefer.detect_one(image, args.object)
        print(f"    elapsed: {time.time() - t0:.1f}s")
        pt = report_point("roborefer", res, width, height)
        if pt:
            px, py = pt

    with Step("3/7  vlm.detect_one  (Molmo)"):
        t0 = time.time()
        res = toolkit.vlm.detect_one(image, args.object)
        print(f"    elapsed: {time.time() - t0:.1f}s")
        pt_vlm = report_point("vlm", res, width, height)
        if pt_vlm and pt:
            # Scale the normalized separation back to pixels; otherwise a real
            # 138 px disagreement would print as "0.1px".
            dist = float(np.hypot((pt[0] - pt_vlm[0]) * width,
                                  (pt[1] - pt_vlm[1]) * height))
            # Not a correctness check -- just a sanity signal. Two independent
            # pointers landing far apart means at least one is wrong, and the
            # image is worth looking at by hand.
            check("roborefer vs vlm agreement", None if dist > 0.25 * width else True,
                  f"{dist:.1f}px apart ({100 * dist / width:.1f}% of image width)")
        if px is None and pt_vlm:
            px, py = pt_vlm

        # ------------------------------------------------------------------
        # 4. index_at is the only vision_ops method the eval keeps enabled.
        # ------------------------------------------------------------------
    with Step("4/7  vision_ops.index_at"):
        if depth_map is not None and px is not None:
            res = toolkit.vision_ops.index_at(depth_map, px, py)
            print(f"    text : {res.text[:200] if res.text else None}")
            d = res.value
            d = float(as_array(d).reshape(-1)[0]) if not isinstance(d, (int, float)) else float(d)
            check("depth at detected point", 0.05 < d < 100.0,
                  f"{d:.3f} m at normalized ({px:.4f}, {py:.4f}) "
                  f"= pixel ({px * width:.0f}, {py * height:.0f})")
        else:
            check("index_at", False, "skipped: no depth map or no point")

        # ------------------------------------------------------------------
        # 5. Segmentation, seeded by the detected point.
        # ------------------------------------------------------------------
    with Step("5/7  sam2.segment_from_point"):
        if px is not None:
            t0 = time.time()
            res = toolkit.sam2.segment_from_point(image, px, py)
            print(f"    elapsed: {time.time() - t0:.1f}s")
            print(f"    text   : {res.text[:200] if res.text else None}")
            v = res.variables or {}
            print(f"    variables: {sorted(v.keys())}")
            for key in ("segmentation_mask", "mask", "masks", "segmentation"):
                if key in v:
                    mask = as_array(v[key])
                    break
            if mask is None and res.value is not None:
                try:
                    mask = as_array(res.value)
                except Exception:
                    mask = None
            if mask is not None:
                m2 = mask if mask.ndim == 2 else mask.reshape(mask.shape[-2:])
                cover = float(m2.astype(bool).mean())
                check("mask shape matches image", m2.shape == (height, width),
                      f"{m2.shape} vs (h={height}, w={width})")
                # An all-zero or near-full mask means the prompt point missed or
                # the model latched onto the background.
                check("mask coverage sane", 0.001 < cover < 0.9,
                      f"{100 * cover:.2f}% of pixels")
                mask = m2
            else:
                check("mask variable", False, "no mask found in result")
        else:
            check("sam2", False, "skipped: no point to seed from")

        # ------------------------------------------------------------------
        # 6. Oriented 3D box. Extents are in metres; implausible extents mean
        #    the point cloud or the mask is in the wrong frame.
        # ------------------------------------------------------------------
    with Step("6/7  bounding_box.compute_bbox"):
        if point_cloud is not None and mask is not None and focal is not None:
            t0 = time.time()
            res = toolkit.bounding_box.compute_bbox(point_cloud, mask, focal)
            print(f"    elapsed: {time.time() - t0:.1f}s")
            print(f"    text   : {res.text[:400] if res.text else None}")
            print(f"    variables: {sorted((res.variables or {}).keys())}")
            check("compute_bbox returned", res.value is not None, f"{type(res.value).__name__}")
        else:
            check("compute_bbox", False, "skipped: missing point cloud, mask or focal length")

        # ------------------------------------------------------------------
        # 7. Grasp generation. The only stochastic tool in the pipeline
        #    (GraspGen is a diffusion model), and the one whose environment
        #    took the most work in P1, so this is the real payoff of the run.
        # ------------------------------------------------------------------
    with Step("7/7  grasp_generator.compute_grasp"):
        if point_cloud is not None and mask is not None and focal is not None:
            t0 = time.time()
            res = toolkit.grasp_generator.compute_grasp(point_cloud, mask, image, focal)
            print(f"    elapsed: {time.time() - t0:.1f}s")
            print(f"    text   : {res.text[:400] if res.text else None}")
            print(f"    variables: {sorted((res.variables or {}).keys())}")
            txt = (res.text or "").lower()
            check("grasps produced", "no collision-free" not in txt and res.value is not None,
                  "see text above")
        else:
            check("compute_grasp", False, "skipped: missing point cloud, mask or focal length")

    section("SUMMARY")
    n_fail = sum(1 for tag, _ in RESULTS if tag == "FAIL")
    n_warn = sum(1 for tag, _ in RESULTS if tag == "WARN")
    for tag, name in RESULTS:
        if tag != "PASS":
            print(f"  {tag}: {name}")
    print(f"\n  {len(RESULTS)} checks: "
          f"{len(RESULTS) - n_fail - n_warn} passed, {n_warn} warned, {n_fail} failed")
    del router
    try:
        shutdown_toolkit()
    except Exception:
        pass
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
