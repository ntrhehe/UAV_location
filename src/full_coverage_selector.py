# -*- coding: utf-8 -*-
"""
full_coverage_selector.py v7
DSM精确footprint + 多边形边界约束 + 空间均匀选择。

通过 pixel_to_cgcs2000 计算每张照片四个角点的真实 lon/lat,
以角点bbox作为footprint, 在ROI多边形内做网格化覆盖选择。

输入:  JPG文件夹 + DSM+DOM 或 DSM+SHP
输出:  选中照片copy + 列表 + GeoJSON + 覆盖图

依赖: rasterio, PIL, matplotlib, numpy
"""

import math, json, shutil, struct, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pixel_to_geo_dem import (
    parse_dji_xmp,
    load_camera_intrinsics,
    load_optimized_pose,
    pixel_to_cgcs2000,
    DSMQuery,
)

M_PER_DEG_LAT = 111319.9


# ---------------------------------------------------------------------------
#  SHP 多边形读取
# ---------------------------------------------------------------------------
def _read_shp_polygon(shp_path, prj_path):
    """读取Polygon(5)或PolygonZ(15)几何。"""
    with open(shp_path, 'rb') as f:
        header = f.read(100)
        file_len = struct.unpack_from('>i', header, 24)[0] * 2
        f.seek(100)
        while f.tell() < file_len:
            rh = f.read(8)
            if len(rh) < 8: break
            rec_len = struct.unpack_from('>i', rh, 4)[0] * 2
            rec = f.read(rec_len)
            if len(rec) < rec_len: break
            st = struct.unpack_from('<i', rec, 0)[0]
            if st in (5, 15):  # Polygon (5) or PolygonZ (15)
                n_parts = struct.unpack_from('<i', rec, 36)[0]
                n_pts = struct.unpack_from('<i', rec, 40)[0]
                parts_off = struct.unpack_from('<' + 'i'*n_parts, rec, 44)
                xy_start = 44 + 4*n_parts
                point_size = 16  # 2 doubles per point
                if st == 15:
                    # PolygonZ: skip Z range + Z array + M range + M array
                    # xy points still start at xy_start, but record is longer
                    # For now just read xy, ignore Z
                    pass
                rings = []
                for pi in range(n_parts):
                    s = parts_off[pi]
                    e = n_pts if pi == n_parts - 1 else parts_off[pi + 1]
                    ring = []
                    for i in range(s, e):
                        off = xy_start + i * point_size
                        x, y = struct.unpack_from('<dd', rec, off)
                        ring.append((x, y))
                    rings.append(ring)
                with open(prj_path, 'r', encoding='utf-8', errors='replace') as pf:
                    crs_wkt = pf.read().strip()
                return rings, crs_wkt
    raise ValueError('No polygon found')


def _transform_rings(rings, src_crs):
    from rasterio.warp import transform
    xs, ys = [], []
    for r in rings:
        for x, y in r:
            xs.append(x); ys.append(y)
    lons, lats = transform(src_crs, 'EPSG:4490', xs, ys)
    result = []
    idx = 0
    for r in rings:
        n = len(r)
        result.append(list(zip(lons[idx:idx+n], lats[idx:idx+n])))
        idx += n
    return result


def _point_in_polygon(lon, lat, rings):
    """射线法判断点是否在多边形内 (只看外环, 忽略洞)."""
    ring = rings[0]  # 外环
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        # 射线向上，检查是否与边相交
        if ((yi > lat) != (yj > lat)):
            x_intersect = (xj - xi) * (lat - yi) / (yj - yi + 1e-20) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _polygon_bbox(poly):
    lons = [p[0] for r in poly for p in r]
    lats = [p[1] for r in poly for p in r]
    return {'lon_min': min(lons), 'lon_max': max(lons),
            'lat_min': min(lats), 'lat_max': max(lats)}


def get_target_from_shp(shp_path):
    shp_path = Path(shp_path)
    prj_path = shp_path.with_suffix('.prj')
    rings, crs = _read_shp_polygon(str(shp_path), str(prj_path))
    poly = _transform_rings(rings, crs)
    bbox = _polygon_bbox(poly)
    return {'bbox': bbox, 'polygon': poly}


def get_target_from_dom(dom_path):
    from rasterio.warp import transform
    import rasterio
    with rasterio.open(str(dom_path)) as ds:
        xm, ym, xM, yM = ds.bounds
        src_crs = ds.crs.to_wkt()
    lons, lats = transform(src_crs, 'EPSG:4490',
                           [xm, xM, xM, xm], [ym, ym, yM, yM])
    bbox = {'lon_min': min(lons), 'lon_max': max(lons),
            'lat_min': min(lats), 'lat_max': max(lats)}
    poly = [[(bbox['lon_min'], bbox['lat_min']),
             (bbox['lon_max'], bbox['lat_min']),
             (bbox['lon_max'], bbox['lat_max']),
             (bbox['lon_min'], bbox['lat_max']),
             (bbox['lon_min'], bbox['lat_min'])]]
    return {'bbox': bbox, 'polygon': poly}


# ---------------------------------------------------------------------------
#  DSM精确footprint (4角点)
# ---------------------------------------------------------------------------
def _m_per_deg_lon_at(lat):
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


# ---- DSM精确footprint (2对角点 + 边缘裁剪) ----
# ---- DSM精确footprint (4角点 + 边缘裁剪) ----
# ---- DSM精确footprint (4角点 + 边缘裁剪) ----
# ---- DSM精确footprint (4角点, 不缩进) ----
# ---- DSM精确footprint (4角点, 缩进10%) ----
def compute_dsm_footprint(jpg_path, dsm, intrinsics, pose, crop_ratio=0.10):
    """用4个缩进角点计算footprint bbox。crop_ratio=0.10表示四边各裁10%。"""
    w, h = pose['img_w'], pose['img_h']
    x0 = int(w * crop_ratio)
    x1 = int(w * (1.0 - crop_ratio)) - 1
    y0 = int(h * crop_ratio)
    y1 = int(h * (1.0 - crop_ratio)) - 1

    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    lons, lats = [], []
    for u, v in corners:
        res = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
        if 'error' not in res:
            lons.append(res['lon'])
            lats.append(res['lat'])

    if len(lons) < 3:
        return None

    return {
        'lon_min': min(lons), 'lon_max': max(lons),
        'lat_min': min(lats), 'lat_max': max(lats),
    }


# ---------------------------------------------------------------------------
#  空间均匀选择 (多边形内)
# ---------------------------------------------------------------------------
def greedy_spatial_selection(footprints, target, grid_size_m=10.0):
    """在target多边形内空间均匀选择照片覆盖。"""
    poly = target['polygon']
    bbox = target['bbox']
    mpl = _m_per_deg_lon_at((bbox['lat_min'] + bbox['lat_max']) / 2.0)
    dlon = grid_size_m / mpl
    dlat = grid_size_m / M_PER_DEG_LAT

    n2f = {fp['name']: fp for fp in footprints}

    # 建立网格索引: 只对poly内的网格点
    cell_photos = {}
    lat = bbox['lat_min'] + dlat / 2.0
    row = 0
    while lat < bbox['lat_max']:
        lon = bbox['lon_min'] + dlon / 2.0
        col = 0
        while lon < bbox['lon_max']:
            if _point_in_polygon(lon, lat, poly):
                key = (col, row)
                cell_photos[key] = set()
                for fp in footprints:
                    b = fp['bbox']
                    if b['lon_min'] <= lon <= b['lon_max'] and \
                       b['lat_min'] <= lat <= b['lat_max']:
                        cell_photos[key].add(fp['name'])
            lon += dlon
            col += 1
        lat += dlat
        row += 1

    if not cell_photos:
        return []

    total_cells = len(cell_photos)

    # 每张照片覆盖的cell数
    photo_cell_count = {}
    for fp in footprints:
        photo_cell_count[fp['name']] = sum(
            1 for cells in cell_photos.values() if fp['name'] in cells
        )

    # 空间均匀选择: 每次挑最稀缺的网格, 选覆盖它最多的照片
    uncovered = set(cell_photos.keys())
    sel_set = set()
    sel = []

    while uncovered:
        rarest, rarest_cnt = None, 999999
        for key in uncovered:
            cnt = len(cell_photos[key] - sel_set)
            if cnt == 0:
                rarest, rarest_cnt = key, 0
                break
            if cnt < rarest_cnt:
                rarest, rarest_cnt = key, cnt

        if rarest is None or rarest_cnt == 0:
            uncovered.discard(rarest)
            continue

        candidates = cell_photos[rarest] - sel_set
        if not candidates:
            uncovered.discard(rarest)
            continue

        best = max(candidates, key=lambda n: photo_cell_count[n])
        sel.append(best)
        sel_set.add(best)

        covered = [k for k in list(uncovered) if best in cell_photos[k]]
        for k in covered:
            uncovered.discard(k)

        if len(sel) % 5 == 0:
            print(f'    [{len(sel)}] covering {len(sel_set)/total_cells*100:.1f}% cells')

    return sel


# ---------------------------------------------------------------------------
#  覆盖图
# ---------------------------------------------------------------------------
def draw_coverage_map(target, all_fps, sel, sel_bboxes, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon as MplPolygon

    poly = target['polygon']
    bbox = target['bbox']
    ref_lat, ref_lon = bbox['lat_min'], bbox['lon_min']
    mpl = _m_per_deg_lon_at((bbox['lat_min'] + bbox['lat_max']) / 2.0)

    def to_xy(lon, lat):
        return (lon - ref_lon) * mpl, (lat - ref_lat) * M_PER_DEG_LAT

    px, py_ = to_xy(bbox['lon_min'], bbox['lat_min'])
    pw, ph = to_xy(bbox['lon_max'], bbox['lat_max'])
    pw -= px; ph -= py_
    fw, fh = max(8, min(22, pw/120)), max(6, min(18, ph/120))

    fig, ax = plt.subplots(figsize=(fw, fh), dpi=150)
    fig.patch.set_facecolor('white')

    for ring in poly:
        verts = [to_xy(lon, lat) for lon, lat in ring]
        ax.add_patch(MplPolygon(verts, fill=False, edgecolor='red',
                                 linewidth=2.5, zorder=10, label='ROI'))

    for fp in all_fps:
        b = fp['bbox']
        x, y = to_xy(b['lon_min'], b['lat_min'])
        w = (b['lon_max'] - b['lon_min']) * mpl
        h = (b['lat_max'] - b['lat_min']) * M_PER_DEG_LAT
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#dddddd',
                                linewidth=0.2, linestyle='--', zorder=1))

    sset = set(sel)
    for b in sel_bboxes:
        x, y = to_xy(b['lon_min'], b['lat_min'])
        w = (b['lon_max'] - b['lon_min']) * mpl
        h = (b['lat_max'] - b['lat_min']) * M_PER_DEG_LAT
        ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor='#2b7cee',
                                alpha=0.30, edgecolor='#1565c0',
                                linewidth=1.0, zorder=5))

    for fp in all_fps:
        if fp['name'] in sset:
            b = fp['bbox']
            x, y = to_xy(b['lon_min'], b['lat_min'])
            w = (b['lon_max'] - b['lon_min']) * mpl
            h = (b['lat_max'] - b['lat_min']) * M_PER_DEG_LAT
            label = fp['name'].split('_')[-1].replace('.JPG', '')
            ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                    fontsize=4.5, color='#0d47a1', weight='bold', zorder=20)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='ROI'),
        Patch(facecolor='#2b7cee', alpha=0.3, edgecolor='#1565c0', label='Selected'),
        Patch(facecolor='none', edgecolor='#cccccc', linestyle='--',
              linewidth=0.5, label='All photos'),
    ], loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_xlabel('East-West (m)'); ax.set_ylabel('North-South (m)')
    ax.set_title('ROI vs Selected Photos (DSM footprint)', fontsize=12, weight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(px - pw*0.05, px + pw*1.05)
    ax.set_ylim(py_ - ph*0.05, py_ + ph*1.05)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'  Map: {out_path}')


# ---------------------------------------------------------------------------
#  输出
# ---------------------------------------------------------------------------
def write_list(sel, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sel) + '\n')


def write_geojson(sel, fps, target, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sset = set(sel)
    feats = []
    if target:
        for ring in target['polygon']:
            feats.append({
                'type': 'Feature',
                'properties': {'name': 'ROI'},
                'geometry': {'type': 'Polygon',
                             'coordinates': [[[lo, la] for lo, la in ring]]},
            })
    for fp in fps:
        b = fp['bbox']
        feats.append({
            'type': 'Feature',
            'properties': {'name': fp['name'], 'selected': fp['name'] in sset},
            'geometry': {'type': 'Polygon', 'coordinates': [[
                [b['lon_min'], b['lat_min']], [b['lon_max'], b['lat_min']],
                [b['lon_max'], b['lat_max']], [b['lon_min'], b['lat_max']],
                [b['lon_min'], b['lat_min']]]]},
        })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats},
                  f, ensure_ascii=False, indent=2)


def copy_selected(sel, src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in sel:
        src = src_dir / name
        if src.exists():
            shutil.copy2(str(src), str(dst_dir / name))


def print_summary(fps, sel, sel_bboxes, target, out_dir, grid_m):
    poly = target['polygon']
    bbox = target['bbox']
    mpl = _m_per_deg_lon_at((bbox['lat_min'] + bbox['lat_max']) / 2.0)
    dlon = grid_m / mpl
    dlat = grid_m / M_PER_DEG_LAT
    total = 0
    covered = 0
    lat = bbox['lat_min'] + dlat / 2.0
    while lat < bbox['lat_max']:
        lon = bbox['lon_min'] + dlon / 2.0
        while lon < bbox['lon_max']:
            if _point_in_polygon(lon, lat, poly):
                total += 1
                for b in sel_bboxes:
                    if b['lon_min'] <= lon <= b['lon_max'] and \
                       b['lat_min'] <= lat <= b['lat_max']:
                        covered += 1
                        break
            lon += dlon
        lat += dlat
    area = total * grid_m * grid_m
    ratio = covered / total if total > 0 else 0

    print()
    print('=' * 60)
    print('  Full Coverage Selector (DSM footprint)')
    print('=' * 60)
    print(f'  Total photos:   {len(fps)}')
    print(f'  Selected:       {len(sel)}')
    print(f'  ROI area:       {area/1e6:.4f} km^2')
    print(f'  Covered:        {area*ratio/1e6:.4f} km^2  ({ratio*100:.1f}%)')
    print(f'  Output:         {out_dir}')
    print()
    for i, n in enumerate(sel, 1):
        fp = next(f for f in fps if f['name'] == n)
        b = fp['bbox']
        w = (b['lon_max'] - b['lon_min']) * _m_per_deg_lon_at((b['lat_min']+b['lat_max'])/2)
        h = (b['lat_max'] - b['lat_min']) * M_PER_DEG_LAT
        print(f'    {i:4d}. {n}  bbox={w:,.0f}x{h:,.0f}m')
    print('=' * 60)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description='Full-coverage selector (DSM footprint)')
    ap.add_argument('input_dir', help='JPG directory')
    ap.add_argument('--dsm', required=True, help='DSM GeoTIFF path')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--dom', default=None)
    ap.add_argument('--shp', default=None)
    ap.add_argument('--cam', default=None)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--grid', type=float, default=10.0)
    args = ap.parse_args()

    if not args.dom and not args.shp:
        print('ERROR: --dom or --shp required.')
        return

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output).resolve() if args.output \
              else input_dir.parent / 'selected'

    print(f'  Input : {input_dir}')
    print(f'  DSM   : {args.dsm}')
    if args.dom: print(f'  DOM   : {args.dom}')
    if args.shp: print(f'  SHP   : {args.shp}')
    print()

    # ---- Load intrinsics ----
    intrinsics = None
    if args.cam:
        print('[0] Loading camera intrinsics ...')
        intrinsics = load_camera_intrinsics(args.cam.split(',')[0].strip())
    else:
        print('[0] Camera intrinsics: from XMP')

    # ---- Load CSV poses ----
    csv_poses = None
    if args.csv:
        print('     Loading CSV poses ...')
        from pixel_to_geo_dem import load_optimized_pose
        csv_path = args.csv
        csv_poses = {}
        with open(csv_path, encoding='gbk', errors='ignore') as f:
            for line in f:
                if '.JPG' not in line.upper():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    csv_poses[Path(parts[0]).name] = {
                        'lat': float(parts[1]), 'lon': float(parts[2]),
                        'alt': float(parts[3]),
                    }

    # ---- Target ----
    print('[1/6] Reading target boundary ...')
    if args.dom:
        target = get_target_from_dom(args.dom)
        label = 'DOM'
    else:
        target = get_target_from_shp(args.shp)
        label = 'SHP'
    b = target['bbox']
    print(f'  {label} bbox: lon {b["lon_min"]:.6f}~{b["lon_max"]:.6f}')
    print(f'               lat {b["lat_min"]:.6f}~{b["lat_max"]:.6f}')
    if 'polygon' in target:
        print(f'               rings={len(target["polygon"])}, '
              f'verts={sum(len(r) for r in target["polygon"])}')
    print()

    # ---- JPG list ----
    print('[2/6] Scanning JPGs ...')
    jpgs = sorted(p for p in input_dir.iterdir()
                  if p.suffix.upper() in ('.JPG', '.JPEG'))
    print(f'  {len(jpgs)} JPGs found')
    if not jpgs:
        return

            # ---- DSM footprints (2对角点 + 边缘裁剪 + GPS粗过滤) ----
    print('[3/6] Computing DSM footprints (2-corners, crop 10%) ...')
    dsm = DSMQuery(args.dsm)
    tb = target['bbox']
    fps = []
    failed = 0
    skipped_gps = 0
    for i, jpg in enumerate(jpgs, 1):
        try:
            if csv_poses and jpg.name in csv_poses:
                pose = parse_dji_xmp(str(jpg))
                pose['lat'] = csv_poses[jpg.name]['lat']
                pose['lon'] = csv_poses[jpg.name]['lon']
                pose['abs_alt'] = csv_poses[jpg.name]['alt']
                pose['roll'] = 0.0
            else:
                pose = parse_dji_xmp(str(jpg))
                pose['roll'] = 0.0

            # GPS粗过滤: 如果照片中心离ROI bbox太远, 跳过
            gps_lat, gps_lon = pose['lat'], pose['lon']
            margin_deg = 0.005  # ~500m
            if (gps_lon > tb['lon_max'] + margin_deg or
                gps_lon < tb['lon_min'] - margin_deg or
                gps_lat > tb['lat_max'] + margin_deg or
                gps_lat < tb['lat_min'] - margin_deg):
                skipped_gps += 1
                continue

            bbox = compute_dsm_footprint(str(jpg), dsm, intrinsics, pose)
            if bbox:
                fps.append({'name': jpg.name, 'bbox': bbox})
            else:
                failed += 1
                if failed <= 3:
                    print(f'  SKIP {jpg.name}: <2 valid corners')
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f'  SKIP {jpg.name}: {e}')
        if i % 100 == 0 or i == len(jpgs):
            t_elapsed = f'{i}/{len(jpgs)}'
            print(f'  {t_elapsed} ...', end='\r')
    dsm.close()
    print(f'  {len(jpgs)} scanned, {len(fps)} OK, {failed} failed, {skipped_gps} skipped (GPS far)')

    if not fps:
        print('  ERROR: no valid footprints.')
        return

    tb = target['bbox']
    in_bbox = []
    for fp in fps:
        b = fp['bbox']
        if b['lon_max'] > tb['lon_min'] and b['lon_min'] < tb['lon_max'] and \
           b['lat_max'] > tb['lat_min'] and b['lat_min'] < tb['lat_max']:
            in_bbox.append(fp)
    print(f'  {len(in_bbox)} within target bbox (of {len(fps)} total)')

    # ---- Select ----
    print('[5/6] Spatial selection ...')
    sel = greedy_spatial_selection(fps, target, args.grid)
    sel_bboxes = [fp['bbox'] for fp in fps if fp['name'] in set(sel)]

    # ---- Output ----
    print('[6/6] Writing output ...')
    copy_selected(sel, input_dir, out_dir)
    write_list(sel, str(out_dir / 'selected_photos.txt'))
    write_geojson(sel, fps, target, str(out_dir / 'coverage_viz.geojson'))
    try:
        draw_coverage_map(target, fps, sel, sel_bboxes,
                          str(out_dir / 'coverage_map.png'))
    except Exception as e:
        print(f'  WARN: map failed: {e}')

    print_summary(fps, sel, sel_bboxes, target, out_dir, args.grid)


if __name__ == '__main__':
    main()