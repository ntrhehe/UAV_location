code = """# -*- coding: utf-8 -*-
import sys, numpy as np, math, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pixel_to_geo_dem import (
    load_camera_intrinsics, parse_dji_xmp, pixel_to_cgcs2000, DSMQuery,
)

NADIR_TOL = 0.5

def is_nadir(pitch):
    return abs(pitch + 90.0) < NADIR_TOL

def parse_csv_poses(csv_path):
    import csv as _csv
    csv_path = Path(csv_path)
    poses = {}
    with open(str(csv_path), encoding='gbk', errors='ignore') as f:
        reader = _csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue
            fp = row[0].strip()
            if '.JPG' not in fp.upper():
                continue
            name = Path(fp).name
            poses[name] = dict(name=name, lat=float(row[1]), lon=float(row[2]),
                               alt=float(row[3]), yaw=float(row[4]),
                               pitch=float(row[5]), roll=float(row[6]))
    return poses

def generate_gcp_grid(pose, dsm, intrinsics, img_w, img_h, grid_step):
    rows = list(range(0, img_h, grid_step))
    cols = list(range(0, img_w, grid_step))
    if not rows or rows[-1] != img_h - 1:
        rows.append(img_h - 1)
    if not cols or cols[-1] != img_w - 1:
        cols.append(img_w - 1)
    from rasterio.control import GroundControlPoint
    gcps = []
    for v in rows:
        for u in cols:
            res = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
            if 'error' not in res:
                gcps.append(GroundControlPoint(row=v, col=u, x=res['lon'], y=res['lat']))
    return gcps

def generate_affine_gcps(pose, dsm, intrinsics, img_w, img_h):
    samples = [
        (0, 0, 'tl'), (img_w - 1, 0, 'tr'),
        (img_w - 1, img_h - 1, 'br'), (0, img_h - 1, 'bl'),
        (img_w // 2, 0, 'tc'), (img_w - 1, img_h // 2, 'rc'),
        (img_w // 2, img_h - 1, 'bc'), (0, img_h // 2, 'lc'),
    ]
    from rasterio.control import GroundControlPoint
    gcps = []
    for u, v, label in samples:
        res = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
        if 'error' not in res:
            gcps.append(GroundControlPoint(row=v, col=u, x=res['lon'], y=res['lat']))
    return gcps

def _write_nadir_affine(jpg_path, pose, dsm, intrinsics, output_path, crs, crop_margin):
    import rasterio, numpy as np
    from rasterio.transform import from_gcps
    from rasterio.warp import reproject, Resampling
    from PIL import Image
    img = Image.open(str(jpg_path))
    arr = np.array(img)
    h, w = arr.shape[:2]
    bands = arr.shape[2] if arr.ndim == 3 else 1
    dtype = arr.dtype
    if crop_margin > 0:
        x0 = int(w * crop_margin); x1 = int(w * (1.0 - crop_margin))
        y0 = int(h * crop_margin); y1 = int(h * (1.0 - crop_margin))
        arr = arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, :]
        hc, wc = arr.shape[:2]
    else:
        x0, y0, hc, wc = 0, 0, h, w
    gcps_raw = generate_affine_gcps(pose, dsm, intrinsics, w, h)
    from rasterio.control import GroundControlPoint
    gcps = []
    for gcp in gcps_raw:
        nc, nr = gcp.col - x0, gcp.row - y0
        if 0 <= nc < wc and 0 <= nr < hc:
            gcps.append(GroundControlPoint(row=nr, col=nc, x=gcp.x, y=gcp.y))
    if len(gcps) < 4:
        print('  FAILED (only {} valid GCPs after crop)'.format(len(gcps)))
        return False
    print('  nadir->affine ({} GCPs, crop={})'.format(len(gcps), crop_margin), end='', flush=True)
    aff = from_gcps(gcps)
    if aff is None:
        print('  FAILED (from_gcps)')
        return False
    corners_px = [(0,0),(wc,0),(wc,hc),(0,hc)]
    lons, lats = [], []
    for c, r in corners_px:
        lon, lat = aff * (c, r)
        lons.append(lon); lats.append(lat)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    gsd_lon, gsd_lat = abs(aff[0]), abs(aff[4])
    if gsd_lon <= 0 or gsd_lat <= 0:
        print('  FAILED (zero GSD)')
        return False
    out_w = max(1, int(np.ceil((lon_max - lon_min) / gsd_lon)))
    out_h = max(1, int(np.ceil((lat_max - lat_min) / gsd_lat)))
    out_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, out_w, out_h)
    out_profile = dict(driver='GTiff', height=out_h, width=out_w, count=bands,
                       dtype=dtype, crs=crs, transform=out_transform, compress='lzw')
    with rasterio.open(str(output_path), 'w', **out_profile) as dst:
        for i in range(bands):
            src_band = arr[:,:,i] if bands > 1 else arr
            dst_band = np.empty((out_h, out_w), dtype=dtype)
            reproject(source=src_band, destination=dst_band,
                      src_transform=aff, src_crs=crs,
                      dst_transform=out_transform, dst_crs=crs,
                      resampling=Resampling.bilinear)
            dst.write(dst_band, i + 1)
    print('  OK')
    return True

def _write_oblique_warp(jpg_path, pose, dsm, intrinsics, output_path, crs, grid_step, crop_margin):
    import rasterio, numpy as np
    from rasterio.warp import reproject, calculate_default_transform
    from rasterio.io import MemoryFile
    from PIL import Image
    img = Image.open(str(jpg_path))
    arr = np.array(img)
    h, w = arr.shape[:2]
    bands = arr.shape[2] if arr.ndim == 3 else 1
    dtype = arr.dtype
    if crop_margin > 0:
        x0 = int(w * crop_margin); x1 = int(w * (1.0 - crop_margin))
        y0 = int(h * crop_margin); y1 = int(h * (1.0 - crop_margin))
        arr = arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, :]
        hc, wc = arr.shape[:2]
    else:
        x0, y0, hc, wc = 0, 0, h, w
    gcps_raw = generate_gcp_grid(pose, dsm, intrinsics, w, h, grid_step)
    from rasterio.control import GroundControlPoint
    gcps = []
    for gcp in gcps_raw:
        nc, nr = gcp.col - x0, gcp.row - y0
        if 0 <= nc < wc and 0 <= nr < hc:
            gcps.append(GroundControlPoint(row=nr, col=nc, x=gcp.x, y=gcp.y))
    if len(gcps) < 4:
        print('  FAILED (only {} valid GCPs after crop)'.format(len(gcps)))
        return False
    n_cols = (w - 1) // grid_step + 2
    n_rows = (h - 1) // grid_step + 2
    print('  oblique GCP: {}x{} -> {} valid, crop={}'.format(n_cols, n_rows, len(gcps), crop_margin), end='', flush=True)
    src_transform = rasterio.transform.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    with MemoryFile() as memfile:
        with memfile.open(driver='GTiff', height=hc, width=wc, count=bands,
                          dtype=dtype, crs=crs, transform=src_transform, gcps=gcps) as mem:
            for i in range(bands):
                mem.write(arr[:,:,i] if bands > 1 else arr, i+1)
        with memfile.open() as src:
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs, crs, src.width, src.height,
                left=src.bounds.left, bottom=src.bounds.bottom,
                right=src.bounds.right, top=src.bounds.top, gcps=src.gcps)
            dst_profile = dict(driver='GTiff', height=dst_h, width=dst_w,
                               count=bands, dtype=dtype, crs=crs,
                               transform=dst_transform, compress='lzw')
            with rasterio.open(str(output_path), 'w', **dst_profile) as dst:
                for i in range(1, bands+1):
                    reproject(source=rasterio.band(src, i),
                              destination=rasterio.band(dst, i),
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=dst_transform, dst_crs=crs,
                              src_gcps=src.gcps,
                              resampling=rasterio.warp.Resampling.bilinear)
    print('  OK')
    return True

def write_geotiff(jpg_path, pose, dsm, intrinsics, output_path,
                  crs='EPSG:4490', grid_step=50, crop_margin=0.10):
    pitch = pose.get('pitch', -90.0)
    if is_nadir(pitch):
        return _write_nadir_affine(jpg_path, pose, dsm, intrinsics, output_path, crs, crop_margin)
    else:
        return _write_oblique_warp(jpg_path, pose, dsm, intrinsics, output_path, crs, grid_step, crop_margin)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Georeference UAV photos to GeoTIFF')
    parser.add_argument('--img', required=True, help='Photo directory')
    parser.add_argument('--dsm', required=True, help='DSM TIFF')
    parser.add_argument('--out', default='./geotiff_output')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--cam', default=None)
    parser.add_argument('--crs', default='EPSG:4490')
    parser.add_argument('--grid-step', type=int, default=500)
    parser.add_argument('--crop-margin', type=float, default=0.10)
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    img_dir = Path(args.img).resolve()
    dsm_path = Path(args.dsm).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print('  img        : {}'.format(img_dir))
    print('  DSM        : {}'.format(dsm_path))
    print('  out        : {}'.format(out_dir))
    print('  crop-margin: {}'.format(args.crop_margin))
    print()
    intrinsics = None
    if args.cam:
        cam_path = Path(args.cam).resolve()
        intrinsics = load_camera_intrinsics(str(cam_path))
        print('[1/4] Intrinsics loaded: F={:.1f}'.format(intrinsics['fx']))
    else:
        print('[1/4] Intrinsics: from XMP')
    all_poses = None
    if args.csv:
        csv_path = Path(args.csv).resolve()
        all_poses = parse_csv_poses(str(csv_path))
        print('[2/4] CSV: {} poses'.format(len(all_poses)))
    else:
        print('[2/4] Poses: from XMP')
    selected = sorted(p.name for p in img_dir.iterdir()
                      if p.suffix.upper() in ('.JPG','.JPEG','.PNG','.TIF','.TIFF'))
    print('  {} images'.format(len(selected)))
    if not selected: return
    print('[ ] Georeferencing ...')
    dsm = DSMQuery(str(dsm_path))
    success, failed = 0, []
    for idx, name in enumerate(selected, 1):
        jpg_path = img_dir / name
        if not jpg_path.exists():
            failed.append((name, 'not found'))
            continue
        if all_poses:
            rec = all_poses.get(name)
            if rec is None:
                failed.append((name, 'no CSV pose'))
                continue
            pose = parse_dji_xmp(str(jpg_path))
            pose['lat'] = rec['lat']; pose['lon'] = rec['lon']
            pose['abs_alt'] = rec['alt']; pose['roll'] = 0.0
        else:
            pose = parse_dji_xmp(str(jpg_path))
            pose['roll'] = 0.0
        print('  [{}/{}] {}'.format(idx, len(selected), name), end='', flush=True)
        out_path = out_dir / (Path(name).stem + '.tif')
        ok = write_geotiff(str(jpg_path), pose, dsm, intrinsics,
                           str(out_path), crs=args.crs, grid_step=args.grid_step,
                           crop_margin=args.crop_margin)
        if ok:
            success += 1
            print('\r  [{}/{}] {}  -> {}'.format(idx, len(selected), name, out_path.name))
        else:
            print('\r  [{}/{}] {}  FAILED'.format(idx, len(selected), name))
            failed.append((name, 'warp failed'))
    dsm.close()
    print()
    print('='*52)
    print('  Success: {}/{}'.format(success, len(selected)))
    if failed:
        print('  Failed:')
        for n, r in failed:
            print('    - {} ({})'.format(n, r))
    print('  Output: {}'.format(out_dir))
    print('='*52)

if __name__ == '__main__':
    main()
"""
with open(r'E:\Users\zhuawawa\Desktop\UAV-location\src\geo_reference.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('geo_reference.py written OK')
