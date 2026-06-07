# -*- coding: utf-8 -*-
'''
pixel_to_geo_dem.py
DJI JPG pixel coordinates -> CGCS2000 geographic coordinates (EPSG:4490).
Uses DSM/DEM TIFF for iterative ray-terrain intersection.

DJI M4E coordinate conventions:
  - Yaw: 0=North, +90=East (clockwise). FlightYaw = drone heading,
    GimbalYaw = gimbal yaw relative to nose.
  - Pitch: 0=horizontal, -90=nadir (straight down).
  - Roll: camera roll around optical axis. XMP GimbalRoll already includes
    the 180 deg physical flip for nadir.
  - Camera frame: X=right(image), Y=down(image), Z=forward(optical axis).
  - World frame: ENU (East, North, Up).

For nadir (pitch ~= -90): effective_yaw = FlightYaw.
For oblique: effective_yaw = FlightYaw + GimbalYaw.

Usage: python pixel_to_geo_dem.py <JPG> <DSM_TIF> <u> <v>
'''

import sys, math, numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
#  geodetic helpers  (WGS-84 / CGCS2000 ellipsoid)
# ---------------------------------------------------------------------------

WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257222101
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = _deg2rad(lat_deg), _deg2rad(lon_deg)
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    X = (N + alt_m) * math.cos(lat) * math.cos(lon)
    Y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    Z = (N * (1.0 - WGS84_E2) + alt_m) * math.sin(lat)
    return np.array([X, Y, Z], dtype=np.float64)


def ecef_to_wgs84(X: float, Y: float, Z: float):
    b   = WGS84_A * math.sqrt(1.0 - WGS84_E2)
    ep2 = (WGS84_A * WGS84_A - b * b) / (b * b)
    lon = math.atan2(Y, X)
    p   = math.sqrt(X * X + Y * Y)
    theta = math.atan2(Z * WGS84_A, p * b)
    st, ct = math.sin(theta), math.cos(theta)
    lat = math.atan2(Z + ep2 * b * st ** 3,
                     p - WGS84_E2 * WGS84_A * ct ** 3)
    sin_lat = math.sin(lat)
    N   = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return (math.degrees(lat), math.degrees(lon), alt)


# ---------------------------------------------------------------------------
#  rotation  camera frame -> ENU
# ---------------------------------------------------------------------------

def make_rotation_dji(absolute_yaw_deg: float,
                      pitch_deg: float,
                      roll_deg: float) -> np.ndarray:
    '''Camera (right, down, forward) -> ENU (East, North, Up).

    Rotation order:  yaw(ENU-Up) -> pitch(camera-right) -> roll(camera-forward).
    '''
    y, p, r = _deg2rad(absolute_yaw_deg), _deg2rad(pitch_deg), _deg2rad(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    fwd_x = sy * cp
    fwd_y = cy * cp
    fwd_z = sp

    hlen = math.hypot(fwd_x, fwd_y)
    if hlen > 1e-10:
        rx0, ry0, rz0 = fwd_y / hlen,  -fwd_x / hlen, 0.0
    else:
        rx0, ry0, rz0 = cy, -sy, 0.0

    dx0 = fwd_y * rz0 - fwd_z * ry0
    dy0 = fwd_z * rx0 - fwd_x * rz0
    dz0 = fwd_x * ry0 - fwd_y * rx0

    rx = cr * rx0 + sr * dx0
    ry = cr * ry0 + sr * dy0
    rz = cr * rz0 + sr * dz0
    dx = -sr * rx0 + cr * dx0
    dy = -sr * ry0 + cr * dy0
    dz = -sr * rz0 + cr * dz0

    return np.column_stack([[rx, ry, rz],
                            [dx, dy, dz],
                            [fwd_x, fwd_y, fwd_z]])
#  XMP parsing
# ---------------------------------------------------------------------------

def parse_dji_xmp(jpg_path):
    '''Extract DJI XMP metadata.'''
    from PIL import Image
    img = Image.open(str(jpg_path))
    xmp = img.getxmp()
    d = xmp['xmpmeta']['RDF']['Description']

    flight_yaw = float(d.get('FlightYawDegree', 0.0))
    gimbal_yaw = float(d.get('GimbalYawDegree', 0.0))
    pitch      = float(d.get('GimbalPitchDegree', -90.0))

    NADIR_TOL = 0.5
    if abs(pitch + 90.0) < NADIR_TOL:
        effective_yaw = flight_yaw
    else:
        effective_yaw = flight_yaw + gimbal_yaw

    return {
        'lat':           float(d['GpsLatitude']),
        'lon':           float(d['GpsLongitude']),
        'abs_alt':       float(d['AbsoluteAltitude']),
        'rel_alt':       float(d.get('RelativeAltitude', 0.0)),
        'flight_yaw':    flight_yaw,
        'gimbal_yaw':    gimbal_yaw,
        'effective_yaw': effective_yaw,
        'pitch':         pitch,
        'roll':          float(d.get('GimbalRollDegree', 0.0)),
        'fx':            float(d.get('CalibratedFocalLength', 3725.0)),
        'fy':            float(d.get('CalibratedFocalLength', 3725.0)),
        'cx':            float(d.get('CalibratedOpticalCenterX', 2640.0)),
        'cy':            float(d.get('CalibratedOpticalCenterY', 1978.0)),
        'img_w':         img.size[0],
        'img_h':         img.size[1],
    }


# ---------------------------------------------------------------------------
#  camera intrinsics loader  (cam.txt from DJI Terra / calibration)
# ---------------------------------------------------------------------------

def load_camera_intrinsics(cam_path):
    '''Load optimised camera intrinsics + distortion from cam.txt.

    Format:
      F:<focal_length> px
      CX:<principal_point_x> px
      CY:<principal_point_y> px
      K1:...
      K2:...
      K3:...
      P1:...
      P2:...
    '''
    params = {}
    with open(str(cam_path), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            val = val.replace('px', '').strip()
            params[key.strip()] = float(val)
    return {
        'fx': params.get('F', 3725.0),
        'fy': params.get('F', 3725.0),
        'cx': params.get('CX', 2640.0),
        'cy': params.get('CY', 1978.0),
        'k1': params.get('K1', 0.0),
        'k2': params.get('K2', 0.0),
        'k3': params.get('K3', 0.0),
        'p1': params.get('P1', 0.0),
        'p2': params.get('P2', 0.0),
    }


def undistort(u: float, v: float, intrinsics: dict):
    '''Undistort pixel coordinates using Brown-Conrady model.

    Args:
        u, v: distorted pixel coordinates
        intrinsics: dict with fx, fy, cx, cy, k1, k2, k3, p1, p2
    Returns:
        (x_undistorted_norm, y_undistorted_norm) in normalized camera coords
    '''
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    k1, k2, k3 = intrinsics['k1'], intrinsics['k2'], intrinsics['k3']
    p1, p2 = intrinsics['p1'], intrinsics['p2']

    # normalised distorted coordinates
    xd = (u - cx) / fx
    yd = (v - cy) / fy

    # fixed-point iteration: undistorted -> distorted (OpenCV convention)
    xu, yu = xd, yd
    for _ in range(8):
        r2 = xu * xu + yu * yu
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        tx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        ty = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        xu = (xd - tx) / radial
        yu = (yd - ty) / radial

    return xu, yu


# ---------------------------------------------------------------------------
#  DSM query  (windowed read to avoid loading huge rasters into RAM)
# ---------------------------------------------------------------------------

class DSMQuery:
    '''Query DSM elevation by (lat, lon) with bilinear interpolation.

    Uses rasterio.warp.transform for EPSG:4490 -> DSM CRS (EPSG:4547).
    Reads only the needed 2x2 pixel window via rasterio windowed read.
    '''

    def __init__(self, tif_path):
        import os, rasterio
        os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
        self.ds = rasterio.open(str(tif_path))
        self.transform = self.ds.transform
        self.h, self.w = self.ds.shape
        self.nodata = self.ds.nodata or -9999.0
        self._crs = self.ds.crs
        self._dtype = self.ds.dtypes[0]

    @property
    def crs(self):
        return self._crs

    def _latlon_to_pixel(self, lat: float, lon: float):
        '''Convert EPSG:4490 (lat, lon) to DSM pixel (col, row).'''
        from rasterio.warp import transform
        xs, ys = transform('EPSG:4490', self._crs, [lon], [lat])
        x, y = xs[0], ys[0]
        col = (x - self.transform.c) / self.transform.a
        row = (y - self.transform.f) / self.transform.e
        return col, row

    def query_elevation(self, lat: float, lon: float) -> float:
        '''Bilinear-interpolated elevation at EPSG:4490 (lat, lon).'''
        col, row = self._latlon_to_pixel(lat, lon)

        c0 = int(math.floor(col))
        r0 = int(math.floor(row))
        c1 = c0 + 1
        r1 = r0 + 1

        row_min = max(r0, 0)
        row_max = min(r1 + 1, self.h)
        col_min = max(c0, 0)
        col_max = min(c1 + 1, self.w)
        if row_max <= row_min or col_max <= col_min:
            return 0.0

        window = ((row_min, row_max), (col_min, col_max))
        data = self.ds.read(1, window=window, boundless=True,
                            fill_value=self.nodata)
        win_h, win_w = data.shape

        def _val(cc: int, rr: int) -> float | None:
            dc = cc - col_min
            dr = rr - row_min
            if 0 <= dc < win_w and 0 <= dr < win_h:
                v = data[dr, dc]
                if v != self.nodata:
                    return float(v)
            return None

        v00 = _val(c0, r0)
        v10 = _val(c1, r0)
        v01 = _val(c0, r1)
        v11 = _val(c1, r1)

        present = [v for v in (v00, v10, v01, v11) if v is not None]
        if len(present) < 2:
            vc = _val(int(round(col)), int(round(row)))
            return vc if vc is not None else 0.0

        mean_v = sum(present) / len(present)
        if v00 is None: v00 = mean_v
        if v10 is None: v10 = mean_v
        if v01 is None: v01 = mean_v
        if v11 is None: v11 = mean_v

        fx = col - c0
        fy = row - r0
        top = v00 * (1.0 - fx) + v10 * fx
        bot = v01 * (1.0 - fx) + v11 * fx
        return top * (1.0 - fy) + bot * fy

    def close(self):
        self.ds.close()


# ---------------------------------------------------------------------------
#  ray 閳?DEM intersection
# ---------------------------------------------------------------------------

def ray_dem_intersection(lat0: float, lon0: float, alt0: float,
                         v_enu: np.ndarray, dsm: DSMQuery,
                         max_dist: float = 8000.0) -> dict:
    '''March a ray through ECEF space until it hits the DSM surface.'''
    lr, lor = _deg2rad(lat0), _deg2rad(lon0)
    slat, clat = math.sin(lr), math.cos(lr)
    slon, clon = math.sin(lor), math.cos(lor)
    dir_ecef = np.array([
        -slon * v_enu[0] - slat * clon * v_enu[1] + clat * clon * v_enu[2],
         clon * v_enu[0] - slat * slon * v_enu[1] + clat * slon * v_enu[2],
                         clat * v_enu[1] + slat * v_enu[2],
    ])
    dir_ecef /= np.linalg.norm(dir_ecef)

    origin = wgs84_to_ecef(lat0, lon0, alt0)
    t      = 0.0
    step   = 3.0

    while t < max_dist:
        t += step
        pt = origin + t * dir_ecef
        lat_pt, lon_pt, alt_pt = ecef_to_wgs84(pt[0], pt[1], pt[2])
        dem_h = dsm.query_elevation(lat_pt, lon_pt) or 0.0
        if alt_pt < dem_h:
            lo, hi = t - step, t
            for _ in range(16):
                tm = (lo + hi) * 0.5
                pm = origin + tm * dir_ecef
                lat_m, lon_m, alt_m = ecef_to_wgs84(pm[0], pm[1], pm[2])
                dem_m = dsm.query_elevation(lat_m, lon_m) or 0.0
                if alt_m < dem_m:
                    hi = tm
                else:
                    lo = tm
            t_final = (lo + hi) * 0.5
            p_final = origin + t_final * dir_ecef
            lat_f, lon_f, alt_f = ecef_to_wgs84(p_final[0], p_final[1],
                                                p_final[2])
            return {'lat': lat_f, 'lon': lon_f, 'alt': alt_f, 'dist': t_final}

    return {'error': f'no intersection within {max_dist} m'}


# ---------------------------------------------------------------------------
#  main conversion
# ---------------------------------------------------------------------------

def pixel_to_cgcs2000(u: float, v: float, pose: dict,
                      dsm: DSMQuery,
                      intrinsics: dict = None) -> dict:
    '''Convert pixel (u, v) to CGCS2000 (lon, lat, alt) via DSM intersection.

    If intrinsics is provided (with distortion coeffs), applies
    Brown-Conrady undistortion before ray tracing.
    '''
    lat0, lon0, alt0 = pose['lat'], pose['lon'], pose['abs_alt']
    yaw   = pose['effective_yaw']
    pitch = pose['pitch']
    roll  = pose['roll']

    if intrinsics:
        # use supplied intrinsics + undistortion
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        xn, yn = undistort(u, v, intrinsics)
    else:
        # fall back to XMP intrinsics (no distortion correction)
        fx, fy = pose['fx'], pose['fy']
        cx, cy = pose['cx'], pose['cy']
        xn = (u - cx) / fx
        yn = (v - cy) / fy

    v_cam = np.array([xn, yn, 1.0])
    v_cam /= np.linalg.norm(v_cam)

    R = make_rotation_dji(yaw, pitch, roll)
    v_enu = R.dot(v_cam)
    v_enu /= np.linalg.norm(v_enu)

    result = ray_dem_intersection(lat0, lon0, alt0, v_enu, dsm)
    if 'error' in result:
        return result
    return {
        'lon': result['lon'],
        'lat': result['lat'],
        'alt': result['alt'],
        'dist_m': result['dist'],
    }


# ---------------------------------------------------------------------------
#  optional: load DJI Terra AT-optimised position
# ---------------------------------------------------------------------------

def load_optimized_pose(jpg_path, pos_csv_path):
    '''Replace camera pose with DJI Terra AT-optimised values from CSV.

    CSV columns (from DJI Terra export):
      鐓х墖鍚嶇О, 绾害, 缁忓害, 楂樺害, Yaw, Pitch, Roll, 姘村钩绮惧害, 鍨傜洿绮惧害
    The lat/lon/alt in the CSV are already bundle-adjusted by DJI Terra.
    '''
    jpg_name = Path(jpg_path).name
    with open(str(pos_csv_path), encoding='gbk', errors='ignore') as f:
        # skip BOM and header (header rows don't contain '.JPG')
        for line in f:
            line = line.lstrip('\ufeff').strip()
            if not line or '.JPG' not in line.upper():
                continue
            if jpg_name in line:
                parts = line.strip().split(',')
                break
        else:
            raise FileNotFoundError(jpg_name)

    # CSV columns: 0=name, 1=lat, 2=lon, 3=alt, 4=yaw, 5=pitch, 6=roll, 7=h_acc, 8=v_acc
    opt_lat = float(parts[1])
    opt_lon = float(parts[2])
    opt_alt = float(parts[3])

    xmp = parse_dji_xmp(jpg_path)
    xmp['lat']     = opt_lat
    xmp['lon']     = opt_lon
    xmp['abs_alt'] = opt_alt
    xmp['roll']    = 0.0   # corrected make_rotation_dji no longer needs gimbal flip compensation
    return xmp


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert DJI image pixel coordinates to CGCS2000 coordinates.')
    parser.add_argument('jpg', help='Path to DJI JPG image')
    parser.add_argument('dsm', help='Path to DSM TIFF file')
    parser.add_argument('u', type=float, help='Pixel column coordinate')
    parser.add_argument('v', type=float, help='Pixel row coordinate')
    parser.add_argument('--csv', help='Path to DJI Terra optimized pose CSV',
                        default=None)
    parser.add_argument('--cam', help='Path to camera intrinsics file (cam.txt)',
                        default=None)
    args = parser.parse_args()

    jpg_path = Path(args.jpg)
    tif_path = Path(args.dsm)
    u, v = args.u, args.v

    print(f'photo: {jpg_path.name}')
    print(f'DSM:   {tif_path.name}')
    print(f'pixel: ({u:.1f}, {v:.1f})')
    print()

    if args.csv:
        print(f'using optimized pose from: {Path(args.csv).name}')
        pose = load_optimized_pose(jpg_path, args.csv)
    else:
        pose = parse_dji_xmp(jpg_path)
        pose['roll'] = 0.0  # corrected rotation matrix
    print(f'camera (EPSG:4490): {pose["lat"]:.8f}, {pose["lon"]:.8f}'
          f'  alt={pose["abs_alt"]:.1f} m')
    print(f'flight yaw={pose["flight_yaw"]:.2f}  '
          f'gimbal yaw={pose["gimbal_yaw"]:.2f}  '
          f'effective yaw={pose["effective_yaw"]:.2f}')
    print(f'gimbal pitch={pose["pitch"]:.2f}  roll={pose["roll"]:.2f}')
    print()

    dsm = DSMQuery(tif_path)
    print(f'DSM CRS: {dsm.crs}')

    intrinsics = None
    if args.cam:
        intrinsics = load_camera_intrinsics(args.cam)
        print(f'using camera intrinsics: F={intrinsics["fx"]:.4f}'
              f'  CX={intrinsics["cx"]:.4f}  CY={intrinsics["cy"]:.4f}')
        print(f'distortion: K1={intrinsics["k1"]:.6f}'
              f'  K2={intrinsics["k2"]:.6f}  K3={intrinsics["k3"]:.6f}'
              f'  P1={intrinsics["p1"]:.6f}  P2={intrinsics["p2"]:.6f}')
    else:
        print(f'using XMP intrinsics: F={pose["fx"]:.4f}'
              f'  CX={pose["cx"]:.4f}  CY={pose["cy"]:.4f}')
    print()

    result = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
    dsm.close()

    if 'error' in result:
        print(f'error: {result["error"]}')
        sys.exit(1)

    print('=== CGCS2000 (EPSG:4490) ===')
    print(f'lon = {result["lon"]:.8f}')
    print(f'lat = {result["lat"]:.8f}')
    print(f'alt = {result["alt"]:.3f} m')
    print(f'dist = {result["dist_m"]:.1f} m')

