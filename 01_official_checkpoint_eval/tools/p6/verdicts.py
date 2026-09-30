#!/usr/bin/env python3
"""P6 人工归类的结果(31+1 条)。每条都记「判成哪一类」和「凭什么」。

判据固定为看图后的三选一,再加上从 trajectory 文本就能定的几类:
  1a 工具错        —— 某个点落在错的物体上,或工具对不同查询返回同一个点
  2a 该调没调      —— 题目需要的量,模型没去取
  2c 参数错        —— 查询串本身错了(拼错物体名)
  3  推理错        —— 拿到了正确且可区分的数据却没用
  3b 坐标系/语义   —— 点都对,但 GT 用的不是图像平面的语义
  4  格式错        —— 答案不在选项集里
  6  标注/指代歧义 —— 点都对、语义也对,GT 或指代本身可疑
  -- 2D 不可分     —— 两点在判定轴上重合,二维投影不含判定所需的信息

生成 p6/manual/verdicts.jsonl。
"""
import json, os

V = [
 # ---- robospatial VQA:above/below,图像 y ≠ 3D 上方 ----
 ("robospatial", 279, "3b", "painting/bed 两点都对;画靠在床头柜上、在床**旁边**不在床**上方**,图像 y 更小只是因为它更远"),
 ("robospatial", 283, "3b", "lamp/bed 两点都对;落地灯立在床边,图像里更高但不在床上方"),
 ("robospatial", 294, "3b", "plastic bag/trash bin 两点都对;袋子在垃圾桶**旁边**的地上,图像 y 更小是因为更靠后"),
 ("robospatial", 304, "3b", "teddy bear/tv 两点都对;熊在电视旁边的架子上,高度相当"),
 ("robospatial", 305, "3b", "teddy bear/vacuum 两点都对;两者并排,间距 0.050"),
 ("robospatial", 307, "3b", "game controller/shelf 两点都对;手柄在左侧柜上、货架在右侧,并非上下关系"),
 ("robospatial", 316, "3b", "chopping boards/fridge 两点都对;砧板在灶台右侧台面,冰箱在左侧,并排不是上下"),
 ("robospatial", 329, "3b", "microwave/bottle 两点都对;同一台面上并排"),
 ("robospatial", 330, "3b", "tissue box/cat tower 两点都对;两件独立家具并排"),
 ("robospatial", 252, "3b", "guitar/lamp 两点都对;吉他靠墙紧挨落地灯,间距 0.035,二维投影几乎重合"),
 ("robospatial", 287, "3b", "grey basket/drawer 两点都对;篮子在地上、抽屉柜更高,GT=no 正确。规则退化(间距 0.016)"),
 # ---- robospatial VQA:left/right,相机系 ≠ benchmark 的参考系 ----
 ("robospatial", 295, "3b", "microwave/fridge 两点都对,相机系下微波炉确实在冰箱左边(0.331 vs 0.898),GT=no ⇒ **参考系不是相机系**"),
 ("robospatial", 310, "3b", "mouse/keyboard 两点都对,相机系下鼠标在键盘右边(0.911 vs 0.511),GT=yes ⇒ 参考系翻转"),
 ("robospatial", 347, "3b", "toilet/shelf 两点都对,相机系下马桶在货架左边(0.215 vs 0.801),GT=yes ⇒ 参考系翻转。间距 0.586,不可能是检测误差"),
 # ---- robospatial VQA:真工具错 ----
 ("robospatial", 273, "1a", "'bottle' 检成了最左的洗手液(0.082),真正的水壶在 0.53 左右;同类多实例歧义"),
 ("robospatial", 274, "1a", "同一张图、同一个错检:'bottle' 又是洗手液。**一次坏检测污染了两个样本**"),
 ("robospatial", 323, "1a", "'tissue' 检到了后方台面上的物体(y=0.089),不是桌面上的目标"),
 ("robospatial", 278, "1a", "'table' 与 'desk' 返回同一张床头柜(0.844,0.592)/(0.839,0.590)——**检测退化**,两个查询同一物体"),
 # ---- robospatial VQA:该调没调 ----
 ("robospatial", 246, "2a", "只 detect_one('paper towel') 一次就作答,**参照物 counter 的位置从未获取**"),
 ("robospatial", 272, "2a", "只检测了 bottle,参照物 water 未检测"),
 ("robospatial", 301, "2a", "只检测了 game controller,参照物 sofa 未检测"),
 # ---- cvb2drelation ----
 ("cvb2drelation", 180, "1a", "'brand name'(0.736,0.372)与 'sign'(0.739,0.372)返回同一点;模型又换了 3 种措辞,全部同一点——**检测退化**"),
 ("cvb2drelation", 621, "1a", "六次不同查询('bottle in red box'/'vase'/'soap in red box'/…)全部返回 (0.558,0.594)。**模型主动尝试了 6 种措辞去绕,绕不过去**"),
 ("cvb2drelation", 169, "2c", "查询串写成 'fluentice tube'(fluorescent 拼错);RoboRefer 照样返回了一个点,错误因此静默传播"),
 ("cvb2drelation", 234, "3",  "模型拿到了可区分的 'building in red box'(0.921,0.648),却用了与 'building' 相同的那个点作答——**取到了正确数据没有用**"),
 ("cvb2drelation", 412, "2D不可分", "person(0.580,0.545)与 cell phone(0.253,0.539)在判定轴(y)上相差 0.006;二维投影不含判定所需信息"),
 ("cvb2drelation", 478, "2D不可分", "person(0.507,0.431)与 bottle(0.503,0.850)在判定轴(x)上相差 0.004;两者竖直叠置"),
 ("cvb2drelation", 263, "6",  "两点都对(红框内植物 0.845,0.18;前景植物 0.855,0.674),图像上红框内的更高 ⇒ 规则给 above,GT=below。指代 'the plants' 究竟指哪一株存疑"),
 # ---- reflocation ----
 ("reflocation", 6,  "1a", "RoboRefer 返回 'Detected 0 instance(s)',模型仍自行给出一个点"),
 ("reflocation", 87, "1a", "题目要「第二近的杯子」,detect_all('cup') 只返回 1 个实例;漏检,且深度序从未获取"),
 # ---- blinkdepth ----
 ("blinkdepth", 69, "4",  "**答案是 'C',而选项只有 (A)/(B)**。模型检测了三个小孩、探了三个深度,然后答了个不存在的选项。`parse_ok` 为真,所以 --strict 没拦住"),
 ("blinkdepth", 86, "1a", "'red dot under label A' 与 'label B' 返回同一点 (0.479,0.681)/(0.479,0.680),两次深度读数逐位相同;模型又调 sam2 重探,仍相同——**检测退化**"),
]

LABEL = {
 "1a": "工具错(检测定位/退化)", "2a": "该调没调", "2c": "参数错(查询串)",
 "3": "推理错", "3b": "坐标系/语义不匹配", "4": "格式错(答案不在选项集)",
 "6": "标注/指代歧义", "2D不可分": "二维投影不可分(信息不足)",
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
    print(f"{len(V)} 条,写入 p6/manual/verdicts.jsonl\n")
    for c, k in sorted(n.items(), key=lambda x: -x[1]):
        print(f"  {k:3}  {LABEL[c]}   {c}")
    print()
    bb = collections.Counter(b for b, _, _, _ in V)
    for b, k in bb.items():
        print(f"  {b:16} {k}")

if __name__ == "__main__":
    main()
