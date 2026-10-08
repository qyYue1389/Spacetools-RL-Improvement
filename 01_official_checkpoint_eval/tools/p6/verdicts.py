#!/usr/bin/env python3
"""Results of the P6 manual classification (31+1 entries). Each entry records "which class it was assigned" and "on what basis".

The criterion is fixed: one of three, chosen after looking at the image, plus a few classes that can be decided from the trajectory text alone:
  1a tool error            — some point lands on the wrong object, or the tool returns the same point for different queries
  2a missing call         — the model never fetched a quantity the question needs
  2c wrong argument       — the query string itself is wrong (misspelled object name)
  3  reasoning error      — got correct, distinguishable data but did not use it
  3b frame/semantics      — the points are all correct, but the GT does not use image-plane semantics
  4  format error         — the answer is not in the option set
  6  label/referent ambiguity — points correct, semantics correct too; the GT or the referent itself is questionable
  -- 2D-indistinguishable — the two points coincide on the deciding axis; the 2D projection lacks the information needed to decide

Generates p6/manual/verdicts.jsonl.
"""
import json, os

V = [
 # ---- robospatial VQA: above/below, image y ≠ 3D "above" ----
 ("robospatial", 279, "3b", "painting/bed both points correct; the painting leans on the nightstand, **beside** the bed, not **above** it; its image y is smaller only because it is farther away"),
 ("robospatial", 283, "3b", "lamp/bed both points correct; the floor lamp stands next to the bed, higher in the image but not above the bed"),
 ("robospatial", 294, "3b", "plastic bag/trash bin both points correct; the bag is on the floor **beside** the bin, its image y is smaller because it is farther back"),
 ("robospatial", 304, "3b", "teddy bear/tv both points correct; the bear is on a shelf next to the TV, at about the same height"),
 ("robospatial", 305, "3b", "teddy bear/vacuum both points correct; the two are side by side, margin 0.050"),
 ("robospatial", 307, "3b", "game controller/shelf both points correct; the controller is on the left cabinet, the shelf is on the right, not an above/below relation"),
 ("robospatial", 316, "3b", "chopping boards/fridge both points correct; the boards are on the counter right of the stove, the fridge is on the left; side by side, not above/below"),
 ("robospatial", 329, "3b", "microwave/bottle both points correct; side by side on the same counter"),
 ("robospatial", 330, "3b", "tissue box/cat tower both points correct; two separate pieces of furniture side by side"),
 ("robospatial", 252, "3b", "guitar/lamp both points correct; the guitar leans on the wall right next to the floor lamp, margin 0.035, nearly coincident in the 2D projection"),
 ("robospatial", 287, "3b", "grey basket/drawer both points correct; the basket is on the floor, the drawer unit is higher, GT=no is correct. Rule degenerates (margin 0.016)"),
 # ---- robospatial VQA: left/right, camera frame ≠ the benchmark's reference frame ----
 ("robospatial", 295, "3b", "microwave/fridge both points correct, in the camera frame the microwave really is left of the fridge (0.331 vs 0.898), GT=no ⇒ **the reference frame is not the camera frame**"),
 ("robospatial", 310, "3b", "mouse/keyboard both points correct, in the camera frame the mouse is right of the keyboard (0.911 vs 0.511), GT=yes ⇒ reference frame flipped"),
 ("robospatial", 347, "3b", "toilet/shelf both points correct, in the camera frame the toilet is left of the shelf (0.215 vs 0.801), GT=yes ⇒ reference frame flipped. Margin 0.586, cannot be a detection error"),
 # ---- robospatial VQA: real tool errors ----
 ("robospatial", 273, "1a", "'bottle' was detected as the leftmost hand soap (0.082), the actual water bottle is around 0.53; ambiguity among multiple instances of the same class"),
 ("robospatial", 274, "1a", "same image, same wrong detection: 'bottle' is the hand soap again. **One bad detection contaminated two samples**"),
 ("robospatial", 323, "1a", "'tissue' detected an object on the counter in the back (y=0.089), not the target on the table"),
 ("robospatial", 278, "1a", "'table' and 'desk' return the same nightstand (0.844,0.592)/(0.839,0.590) — **detection degenerates**, two queries, one object"),
 # ---- robospatial VQA: missing call ----
 ("robospatial", 246, "2a", "called detect_one('paper towel') once and answered, **the position of the reference object counter was never fetched**"),
 ("robospatial", 272, "2a", "only bottle was detected, the reference object water was not detected"),
 ("robospatial", 301, "2a", "only game controller was detected, the reference object sofa was not detected"),
 # ---- cvb2drelation ----
 ("cvb2drelation", 180, "1a", "'brand name' (0.736,0.372) and 'sign' (0.739,0.372) return the same point; the model tried 3 more phrasings, all the same point — **detection degenerates**"),
 ("cvb2drelation", 621, "1a", "six different queries ('bottle in red box'/'vase'/'soap in red box'/…) all return (0.558,0.594). **The model actively tried 6 phrasings to work around it and could not**"),
 ("cvb2drelation", 169, "2c", "the query string was written as 'fluentice tube' (fluorescent misspelled); RoboRefer still returned a point, so the error propagated silently"),
 ("cvb2drelation", 234, "3",  "the model got the distinguishable 'building in red box' (0.921,0.648) but answered with the same point as 'building' — **fetched the correct data and did not use it**"),
 ("cvb2drelation", 412, "2D-indistinguishable", "person (0.580,0.545) and cell phone (0.253,0.539) differ by 0.006 on the deciding axis (y); the 2D projection lacks the information needed to decide"),
 ("cvb2drelation", 478, "2D-indistinguishable", "person (0.507,0.431) and bottle (0.503,0.850) differ by 0.004 on the deciding axis (x); the two are stacked vertically"),
 ("cvb2drelation", 263, "6",  "both points correct (plant in red box 0.845,0.18; foreground plant 0.855,0.674), the one in the red box is higher in the image ⇒ rule gives above, GT=below. Which plant the referent 'the plants' means is questionable"),
 # ---- reflocation ----
 ("reflocation", 6,  "1a", "RoboRefer returns 'Detected 0 instance(s)', the model still gives a point on its own"),
 ("reflocation", 87, "1a", "the question asks for \"the second-closest cup\", detect_all('cup') returns only 1 instance; missed detection, and the depth order was never fetched"),
 # ---- blinkdepth ----
 ("blinkdepth", 69, "4",  "**the answer is 'C', but the only options are (A)/(B)**. The model detected three children, probed three depths, then answered with an option that does not exist. `parse_ok` is true, so --strict did not catch it"),
 ("blinkdepth", 86, "1a", "'red dot under label A' and 'label B' return the same point (0.479,0.681)/(0.479,0.680), the two depth readings are bit-for-bit identical; the model called sam2 to re-probe, still the same — **detection degenerates**"),
]

LABEL = {
 "1a": "tool error (detection localization/degeneration)", "2a": "missing call", "2c": "wrong argument (query string)",
 "3": "reasoning error", "3b": "frame/semantics mismatch", "4": "format error (answer not in option set)",
 "6": "label/referent ambiguity", "2D-indistinguishable": "2D projection indistinguishable (insufficient information)",
}

def main():
    os.makedirs("p6/manual", exist_ok=True)
    with open("p6/manual/verdicts.jsonl", "w", encoding="utf-8") as f:
        for b, s, c, why in V:
            f.write(json.dumps({"benchmark": b, "sample_id": s, "class": c,
                                "class_label": LABEL[c], "evidence": why},
                               ensure_ascii=False) + "\n")
    import collections
    n = collections.Counter(c for _, _, c, _ in V)
    print(f"{len(V)} entries, written to p6/manual/verdicts.jsonl\n")
    for c, k in sorted(n.items(), key=lambda x: -x[1]):
        print(f"  {k:3}  {LABEL[c]}   {c}")
    print()
    bb = collections.Counter(b for b, _, _, _ in V)
    for b, k in bb.items():
        print(f"  {b:16} {k}")

if __name__ == "__main__":
    main()
