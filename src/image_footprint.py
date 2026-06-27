# -*- coding: utf-8 -*-
'''
image_footprint.py
Compute the geographic footprint (bounding box) of a DJI image using DSM.
Projects all four corners + edge midpoints to CGCS2000 (EPSG:4490).
九个点采样DSM 查询
Usage:
    python image_footprint.py <JPG> <DSM_TIF> [--cam cam.txt] [--csv pos.csv]
'''

import sys, math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pixel_to_geo_dem import (
    parse_dji_xmp, load_camera_intrinsics, load_optimized_pose,
    pixel_to_cgcs2000, DSMQuery,
)


def compute_footprint(jpg_path: Path, tif_path: Path,
                      intrinsics: dict = None,
                      pose: dict = None) -> dict:
    '''Project image corners + edge midpoints to geographic coordinates.

    Returns dict with:
      - corners:   [tl, tr, br, bl]  (top-left, top-right, bottom-right, bottom-left)
      - midpoints: [top, right, bottom, left]
      - bbox:      {lon_min, lon_max, lat_min, lat_max, lon_cen, lat_cen}
      - img_w, img_h
      - lon_span, lat_span (approx meters)
    '''
    dsm = DSMQuery(tif_path)

    w = pose['img_w']
    h = pose['img_h']

    # 9 sample points: 4 corners + 4 edge midpoints + image center
    samples = [
        (0,     0,     'tl'),       # top-left
        (w//2,  0,     'tc'),       # top-center
        (w-1,   0,     'tr'),       # top-right
        (w-1,  h//2,   'rc'),       # right-center
        (w-1,  h-1,    'br'),       # bottom-right
        (w//2, h-1,    'bc'),       # bottom-center
        (0,    h-1,    'bl'),       # bottom-left
        (0,    h//2,   'lc'),       # left-center
        (w//2, h//2,   'cc'),       # image center
    ]

    points = {}
    valid_lons, valid_lats = [], []

    for u, v, label in samples:
        result = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
        if 'error' in result:
            print(f'  {label:>4s} ({u:5.0f}, {v:5.0f})  ERROR: {result["error"]}')
            continue
        points[label] = {
            'u': u, 'v': v,
            'lon': result['lon'],
            'lat': result['lat'],
            'alt': result['alt'],
            'dist': result['dist_m'],
        }
        valid_lons.append(result['lon'])
        valid_lats.append(result['lat'])

    dsm.close()

    if not points:
        return {'error': 'no valid ground intersections'}

    # bounding box
    lon_min, lon_max = min(valid_lons), max(valid_lons)
    lat_min, lat_max = min(valid_lats), max(valid_lats)
    lon_cen = (lon_min + lon_max) / 2
    lat_cen = (lat_min + lat_max) / 2

    # approximate meter span
    mid_lat_rad = math.radians(lat_cen)
    m_per_deg_lat = 111319.9
    m_per_deg_lon = 111319.9 * math.cos(mid_lat_rad)
    lon_span_m = (lon_max - lon_min) * m_per_deg_lon
    lat_span_m = (lat_max - lat_min) * m_per_deg_lat

    return {
        'corners': {
            'tl': points.get('tl'),
            'tr': points.get('tr'),
            'br': points.get('br'),
            'bl': points.get('bl'),
        },
        'midpoints': {
            'top':    points.get('tc'),
            'right':  points.get('rc'),
            'bottom': points.get('bc'),
            'left':   points.get('lc'),
        },
        'center': points.get('cc'),
        'bbox': {
            'lon_min':  lon_min,
            'lon_max':  lon_max,
            'lat_min':  lat_min,
            'lat_max':  lat_max,
            'lon_cen':  lon_cen,
            'lat_cen':  lat_cen,
            'lon_span_m': lon_span_m,
            'lat_span_m': lat_span_m,
        },
        'img_w': w,
        'img_h': h,
    }


def print_report(footprint: dict):
    '''Print human-readable footprint report.'''
    if 'error' in footprint:
        print(f'error: {footprint["error"]}')
        return

    b = footprint['bbox']
    print('=' * 60)
    print('  Geographic Footprint  (CGCS2000 / EPSG:4490)')
    print('=' * 60)
    print(f'  image size  : {footprint["img_w"]} x {footprint["img_h"]}')
    print()
    print(f'  lon range   : {b["lon_min"]:.8f}  ~  {b["lon_max"]:.8f}')
    print(f'  lat range   : {b["lat_min"]:.8f}  ~  {b["lat_max"]:.8f}')
    print(f'  lon span    : {b["lon_span_m"]:.1f} m')
    print(f'  lat span    : {b["lat_span_m"]:.1f} m')
    print(f'  center      : {b["lon_cen"]:.8f}, {b["lat_cen"]:.8f}')
    print()

    def _print_pt(label, pt):
        if pt:
            print(f'  {label:>6s}  ({pt["u"]:5.0f}, {pt["v"]:5.0f})'
                  f'  →  {pt["lon"]:.8f}  {pt["lat"]:.8f}'
                  f'  alt={pt["alt"]:.1f}m  dist={pt["dist"]:.1f}m')
        else:
            print(f'  {label:>6s}  (no intersection)')

    print('  --- corners ---')
    _print_pt('TL', footprint['corners']['tl'])
    _print_pt('TR', footprint['corners']['tr'])
    _print_pt('BR', footprint['corners']['br'])
    _print_pt('BL', footprint['corners']['bl'])
    print()
    print('  --- midpoints ---')
    _print_pt('TOP',    footprint['midpoints']['top'])
    _print_pt('RIGHT',  footprint['midpoints']['right'])
    _print_pt('BOTTOM', footprint['midpoints']['bottom'])
    _print_pt('LEFT',   footprint['midpoints']['left'])
    print()
    print('  --- image center ---')
    _print_pt('CENTER', footprint['center'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute DJI image geographic footprint.')
    parser.add_argument('jpg', help='Path to DJI JPG image')
    parser.add_argument('dsm', help='Path to DSM TIFF file')
    parser.add_argument('--csv', help='Path to DJI Terra optimized pose CSV',
                        default=None)
    parser.add_argument('--cam', help='Path to camera intrinsics (cam.txt)',
                        default=None)
    args = parser.parse_args()

    jpg_path = Path(args.jpg)
    tif_path = Path(args.dsm)

    print(f'photo : {jpg_path.name}')
    print(f'DSM   : {tif_path.name}')
    print()

    # load pose
    if args.csv:
        pose = load_optimized_pose(jpg_path, args.csv)
    else:
        pose = parse_dji_xmp(jpg_path)
        pose['roll'] = 0.0  # corrected rotation matrix

    print(f'camera: {pose["lat"]:.8f}, {pose["lon"]:.8f}  alt={pose["abs_alt"]:.1f}m')
    print(f'ypr   : yaw={pose["effective_yaw"]:.2f}  pitch={pose["pitch"]:.2f}  roll={pose["roll"]:.2f}')
    print()

    # load intrinsics
    intrinsics = None
    if args.cam:
        intrinsics = load_camera_intrinsics(args.cam)
        print(f'intrinsics: F={intrinsics["fx"]:.2f}  CX={intrinsics["cx"]:.2f}  CY={intrinsics["cy"]:.2f}')
    else:
        print(f'intrinsics: F={pose["fx"]:.2f}  CX={pose["cx"]:.2f}  CY={pose["cy"]:.2f} (XMP)')
    print()

    footprint = compute_footprint(jpg_path, tif_path, intrinsics, pose)
    print_report(footprint)
