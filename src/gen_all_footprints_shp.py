# -*- coding: utf-8 -*-
"""gen_all_footprints_shp.py - 遍历JPG目录,按名称编号排序+抽稀,DSM footprint,输出SHP+复制图片.

航向抽稀策略：照片航向重叠率高(70%), 按文件名编号每隔N张取1张,
可大幅减少冗余的同时保证覆盖。配合后续每张裁剪边缘10%减轻变形影响。
所有航线全保留, 保证旁向边缘无缝隙。

Usage:
    python gen_all_footprints_shp.py <JPG_DIR> <DSM_TIF> <ROI_SHP> <OUT_SHP>
                                     [--step 5] [--copy_dir <DIR>]
"""
import sys, os, re, shutil, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from full_coverage_selector import get_target_from_shp, compute_dsm_footprint, _point_in_polygon
from pixel_to_geo_dem import DSMQuery
import fiona
from fiona.crs import from_epsg

NS_DJI = "http://www.dji.com/drone-dji/1.0/"


def _extract_sort_key(name):
    """从文件名提取末尾编号, 如 DJI_..._0001_V.JPG -> 1"""
    base = os.path.splitext(name)[0]
    parts = base.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return 0


def parse_xmp_direct(jpg_path):
    with open(jpg_path, "rb") as f:
        data = f.read()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        raise ValueError("No XMP")
    xmp_str = data[start:end + 13].decode("utf-8", errors="replace")
    ns = {"x": "adobe:ns:meta/",
          "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
          "drone-dji": NS_DJI}
    root = ET.fromstring(xmp_str)
    desc = root.find(".//rdf:Description", ns)
    if desc is None:
        raise ValueError("No Description")

    def a(name):
        return float(desc.attrib.get("{" + NS_DJI + "}" + name, "0"))

    fy = a("FlightYawDegree")
    gy = a("GimbalYawDegree")
    p = a("GimbalPitchDegree")
    ey = fy if abs(p + 90.0) < 0.5 else fy + gy
    return {
        "lat": a("GpsLatitude"),
        "lon": a("GpsLongitude"),
        "abs_alt": a("AbsoluteAltitude"),
        "rel_alt": a("RelativeAltitude"),
        "flight_yaw": fy,
        "gimbal_yaw": gy,
        "pitch": p,
        "roll": a("GimbalRollDegree"),
        "effective_yaw": ey,
        "fx": a("CalibratedFocalLength"),
        "fy": a("CalibratedFocalLength"),
        "cx": a("CalibratedOpticalCenterX"),
        "cy": a("CalibratedOpticalCenterY"),
        "img_w": 5280,
        "img_h": 3956,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="遍历JPG, 按编号排序抽稀, DSM footprint, 输出SHP+复制图片")
    ap.add_argument("jpg_dir", help="JPG 目录")
    ap.add_argument("dsm_path", help="DSM GeoTIFF 路径")
    ap.add_argument("roi_path", help="ROI SHP 路径")
    ap.add_argument("out_shp", help="输出 SHP 路径")
    ap.add_argument("--step", type=int, default=1,
                    help="每隔N张取一张, 默认1=全部")
    ap.add_argument("--copy_dir", default=None,
                    help="将选中的JPG复制到此目录")
    args = ap.parse_args()

    jpg_dir = args.jpg_dir
    dsm_path = args.dsm_path
    roi_path = args.roi_path
    out_shp = args.out_shp
    step = args.step
    copy_dir = args.copy_dir

    # ---- 1. ROI ----
    print("[1/6] 读取 ROI ...")
    target = get_target_from_shp(roi_path)
    tb = target["bbox"]
    poly = target["polygon"]
    print("  ROI bbox: lon %.6f~%.6f, lat %.6f~%.6f" %
          (tb["lon_min"], tb["lon_max"], tb["lat_min"], tb["lat_max"]))

    # ---- 2. 扫描JPG + 按末尾编号排序 + 抽稀 ----
    print("[2/6] 扫描JPG, 按编号排序, 每隔%d张取一张 ..." % step)
    all_jpgs = [p for p in os.listdir(jpg_dir) if p.upper().endswith(".JPG")]
    all_jpgs.sort(key=_extract_sort_key)
    print("  原始 %d 张 JPG" % len(all_jpgs))

    # 抽稀: 索引 0, step, 2*step, 3*step ...
    sampled = [all_jpgs[i] for i in range(0, len(all_jpgs), step)]
    print("  抽稀后 %d 张 (step=%d)" % (len(sampled), step))
    if step > 1:
        print("  前5张抽稀结果:")
        for s in sampled[:5]:
            print("    %s  (编号 %d)" % (s, _extract_sort_key(s)))

    # ---- 3. 计算 Footprint ----
    print("[3/6] 计算 DSM footprint (仅ROI内) ...")
    dsm = DSMQuery(dsm_path)
    margin_deg = 0.005
    results = []
    failed = 0
    skipped = 0
    for i, fn in enumerate(sampled, 1):
        jpg_full = os.path.join(jpg_dir, fn)
        try:
            pose = parse_xmp_direct(jpg_full)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print("  FAIL parse: %s (%s)" % (fn, e))
            continue
        if (pose["lon"] > tb["lon_max"] + margin_deg or
            pose["lon"] < tb["lon_min"] - margin_deg or
            pose["lat"] > tb["lat_max"] + margin_deg or
            pose["lat"] < tb["lat_min"] - margin_deg):
            skipped += 1
            continue
        if not _point_in_polygon(pose["lon"], pose["lat"], poly):
            skipped += 1
            continue
        try:
            bbox = compute_dsm_footprint(jpg_full, dsm, None, pose)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print("  FAIL footprint: %s (%s)" % (fn, e))
            continue
        if bbox is None:
            failed += 1
            if failed <= 3:
                print("  FAIL footprint: %s (<3 corners)" % fn)
            continue
        results.append({
            "name": fn,
            "gps_lon": pose["lon"],
            "gps_lat": pose["lat"],
            "bbox": bbox,
        })
        if i % 50 == 0 or i == len(sampled):
            print("  %d/%d ... %d OK" % (i, len(sampled), len(results)))
    dsm.close()
    print("  Done: %d OK, %d failed, %d skipped (outside ROI)" %
          (len(results), failed, skipped))
    if not results:
        print("  ERROR: no valid footprints!")
        sys.exit(1)

    # ---- 4. 复制选中照片 ----
    if copy_dir:
        print("[4/6] 复制选中JPG到 %s ..." % copy_dir)
        os.makedirs(copy_dir, exist_ok=True)
        for r in results:
            shutil.copy2(os.path.join(jpg_dir, r["name"]),
                         os.path.join(copy_dir, r["name"]))
        print("  已复制 %d 张" % len(results))

    # ---- 5. 输出SHP ----
    print("[5/6] 写入 SHP ...")
    schema = {
        "geometry": "Polygon",
        "properties": {"name": "str", "gps_lon": "float", "gps_lat": "float"},
    }
    with fiona.open(out_shp, "w", driver="ESRI Shapefile",
                    schema=schema, crs=from_epsg(4490)) as dst:
        for r in results:
            b = r["bbox"]
            ring = [[(b["lon_min"], b["lat_min"]),
                     (b["lon_max"], b["lat_min"]),
                     (b["lon_max"], b["lat_max"]),
                     (b["lon_min"], b["lat_max"]),
                     (b["lon_min"], b["lat_min"])]]
            dst.write({
                "geometry": {"type": "Polygon", "coordinates": ring},
                "properties": {
                    "name": r["name"],
                    "gps_lon": r["gps_lon"],
                    "gps_lat": r["gps_lat"],
                },
            })
    print("  SHP: %s  Records: %d" % (out_shp, len(results)))

    # ---- 6. 总结 ----
    print("[6/6] 完成!")
    print("  原始: %d | 抽稀后: %d (step=%d) | ROI内有效: %d" %
          (len(all_jpgs), len(sampled), step, len(results)))
    print("  失败: %d | 跳过(ROI外): %d" % (failed, skipped))
    if copy_dir:
        print("  选中照片已复制到: %s" % copy_dir)


if __name__ == "__main__":
    main()
