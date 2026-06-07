# -*- coding: utf-8 -*-
"""
Labelme JSON -> YOLO txt

Usage:  Modify IMG_DIR and CLASS_MAP below, then run:
        py json2yolo.py

Dir layout:
    IMG_DIR/
    +-- *.json          # labelme annotation files
    +-- images/         # images (optional, dir just needs to exist)
    +-- labels/         # output YOLO txt (auto-created)
"""

import json, os, io

# ============ CONFIG ============
IMG_DIR = r"test\img"

CLASS_MAP = {
    "diangan": 0,
}
# ================================


def convert(img_dir, class_map):
    labels_dir = os.path.join(img_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    converted = 0
    skipped = 0

    for fname in os.listdir(img_dir):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(img_dir, fname)

        with io.open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        lines = []
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in class_map:
                print("  [SKIP] unknown label '{0}' in {1}".format(label, fname))
                skipped += 1
                continue
            cid = class_map[label]
            pts = shape["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            lines.append("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(cid, cx, cy, w, h))

        base = fname[:-5]
        txt_name = base + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        with io.open(txt_path, "w", encoding="utf-8") as f:
            f.write(u"\n".join(lines))

        converted += 1
        print("  {0} -> labels/{1} ({2} boxes)".format(fname, txt_name, len(lines)))

    print("\nDone! {0} files converted, {1} shapes skipped.".format(converted, skipped))


if __name__ == "__main__":
    convert(IMG_DIR, CLASS_MAP)
