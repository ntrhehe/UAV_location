import json, math, numpy as np
from pathlib import Path

def dr(d):
    return d * math.pi / 180.0

def rd(r):
    return r * 180.0 / math.pi

E = {"a": 6378137.0, "f": 1.0 / 298.257222101}
E["e2"] = 2 * E["f"] - E["f"] ** 2

def wgs84_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = dr(lat_deg), dr(lon_deg)
    a, e2 = E["a"], E["e2"]
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    X = (N + alt_m) * math.cos(lat) * math.cos(lon)
    Y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    Z = (N * (1.0 - e2) + alt_m) * math.sin(lat)
    return np.array([X, Y, Z])

def ecef_to_wgs84(X, Y, Z):
    a, e2 = E["a"], E["e2"]
    b = a * math.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    lon = math.atan2(Y, X)
    p = math.sqrt(X * X + Y * Y)
    theta = math.atan2(Z * a, p * b)
    st = math.sin(theta)
    ct = math.cos(theta)
    lat = math.atan2(Z + ep2 * b * st**3, p - e2 * a * ct**3)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return (rd(lat), rd(lon), alt)

def enu_to_ecef(enu, ref_lat, ref_lon, ref_alt):
    ref = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)
    lat, lon = dr(ref_lat), dr(ref_lon)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    e, n, u = enu[0], enu[1], enu[2]
    X = ref[0] - slon*e - slat*clon*n + clat*clon*u
    Y = ref[1] + clon*e - slat*slon*n + clat*slon*u
    Z = ref[2] + clat*n + slat*u
    return np.array([X, Y, Z])

def make_rotation_dji(yaw, pitch, roll):
    '''Camera (right, down, forward) -> ENU (East, North, Up).

    Yaw: 0=North, +90=East. Pitch: 0=horizontal, -90=nadir.
    Rotation order: yaw(Up) -> pitch(right) -> roll(forward).
    '''
    y, p, r = dr(yaw), dr(pitch), dr(roll)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    # forward vector (optical axis in ENU)
    fwd_x = sy * cp
    fwd_y = cy * cp
    fwd_z = sp

    # right vector (perpendicular to forward, horizontal component rotated by roll)
    hlen = math.hypot(fwd_x, fwd_y)
    if hlen > 1e-10:
        rx0, ry0, rz0 = -fwd_y / hlen,  fwd_x / hlen, 0.0
    else:
        rx0, ry0, rz0 = 1.0, 0.0, 0.0

    # down vector = forward x right (cross product)
    dx0 = fwd_y * rz0 - fwd_z * ry0
    dy0 = fwd_z * rx0 - fwd_x * rz0
    dz0 = fwd_x * ry0 - fwd_y * rx0

    # apply roll around forward axis
    rx = cr * rx0 + sr * dx0
    ry = cr * ry0 + sr * dy0
    rz = cr * rz0 + sr * dz0
    dx = -sr * rx0 + cr * dx0
    dy = -sr * ry0 + cr * dy0
    dz = -sr * rz0 + cr * dz0

    return np.column_stack([[rx, ry, rz],
                            [dx, dy, dz],
                            [fwd_x, fwd_y, fwd_z]])

def pixel_to_geo(u, v, img, cam, dem_query=None):
    lat0, lon0, alt0 = img["position"]
    yaw = img["orientation"][0]
    pitch = img["orientation"][1]
    roll = img["orientation"][2]
    rel_h = img.get("relative_height", 0)
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    v_cam = np.array([x_norm, y_norm, 1.0])
    v_cam = v_cam / np.linalg.norm(v_cam)
    R = make_rotation_dji(yaw, pitch, roll)
    v_enu = R @ v_cam
    v_enu = v_enu / np.linalg.norm(v_enu)
    ray_origin = wgs84_to_ecef(lat0, lon0, alt0)
    if dem_query:
        result = _ray_dem(ray_origin, v_enu, lat0, lon0, alt0, dem_query)
        result["method"] = "dem"
    else:
        ground_elev = alt0 - rel_h
        result = _ray_flat(ray_origin, v_enu, lat0, lon0, alt0, ground_elev)
        result["method"] = "flat_ground"
    result["cam_lat"] = lat0
    result["cam_lon"] = lon0
    result["cam_alt"] = alt0
    result["ray_enu"] = v_enu.tolist()
    return result

def _ray_flat(origin_ecef, dir_enu, ref_lat, ref_lon, ref_alt, ground_elev):
    ground_u = ground_elev - ref_alt
    if abs(dir_enu[2]) < 1e-10:
        return {"error": "ray parallel to ground"}
    t = ground_u / dir_enu[2]
    if t <= 0:
        return {"error": "intersection behind camera"}
    enu_pt = np.array([t*dir_enu[0], t*dir_enu[1], t*dir_enu[2]])
    ecef_pt = enu_to_ecef(enu_pt, ref_lat, ref_lon, ref_alt)
    lat, lon, alt = ecef_to_wgs84(ecef_pt[0], ecef_pt[1], ecef_pt[2])
    return {"lat": lat, "lon": lon, "alt": alt, "t": t}

def _ray_dem(origin, dir_enu, ref_lat, ref_lon, ref_alt, dem):
    lr = dr(ref_lat)
    lor = dr(ref_lon)
    sla, cla = math.sin(lr), math.cos(lr)
    slo, clo = math.sin(lor), math.cos(lor)
    dir_ecef = np.array([
        -slo*dir_enu[0] - sla*clo*dir_enu[1] + cla*clo*dir_enu[2],
         clo*dir_enu[0] - sla*slo*dir_enu[1] + cla*slo*dir_enu[2],
                         cla*dir_enu[1] + sla*dir_enu[2]
    ])
    dir_ecef = dir_ecef / np.linalg.norm(dir_ecef)
    t = 0.0
    step = 10.0
    while t < 5000:
        t += step
        pt = origin + t * dir_ecef
        lat, lon, alt_ray = ecef_to_wgs84(pt[0], pt[1], pt[2])
        dem_alt = dem(lat, lon)
        if alt_ray < dem_alt:
            tl, th = t - step, t
            for _ in range(12):
                tm = (tl + th) / 2
                pm = origin + tm * dir_ecef
                latm, lonm, altm = ecef_to_wgs84(pm[0], pm[1], pm[2])
                demm = dem(latm, lonm)
                if altm < demm:
                    th = tm
                else:
                    tl = tm
            tf = (tl + th) / 2
            pf = origin + tf * dir_ecef
            latf, lonf, altf = ecef_to_wgs84(pf[0], pf[1], pf[2])
            return {"lat": latf, "lon": lonf, "alt": altf, "t": tf}
    return {"error": "no intersection"}


if __name__ == "__main__":
    PD = r"E:\Users\zhuawawa\Documents\DJI\DJITerra\18608448201\qk1-2"
    IL = PD + r"\images\survey\image_list.json"
    SR = PD + r"\AT\report\sfm_report.json"

    with open(IL, "r", encoding="utf-8") as f:
        il = json.load(f)
    with open(SR, "r", encoding="utf-8") as f:
        sfm = json.load(f)
    cam = sfm["cameras"][0]["optimised cameras"][0]["optimised camera param"]

    print(f"{len(il)} photos, K: fx={cam['fx']:.2f} fy={cam['fy']:.2f} cx={cam['cx']:.2f} cy={cam['cy']:.2f}")

    # test first photo
    img = il[0]
    lat0, lon0, alt0 = img["position"]
    ypr = img["orientation"]
    rh = img["relative_height"]
    name = Path(img["path"]).name
    print(f"\n{name}")
    print(f"  pos: ({lat0:.8f}, {lon0:.8f}, {alt0:.3f})")
    print(f"  ypr: {ypr}  rel_h: {rh:.1f}m")

    # image center: should fall near camera position
    u, v = cam["cx"], cam["cy"]
    print(f"\n--- Image center ({u:.0f}, {v:.0f}) ---")
    r = pixel_to_geo(u, v, img, cam)
    if r.get("error"):
        print(f"ERROR: {r['error']}")
    else:
        print(f"  lat={r['lat']:.8f} lon={r['lon']:.8f} alt={r['alt']:.3f}m t={r['t']:.1f}m")
        dlat = abs(r["lat"] - lat0) * 111319.9
        dlon = abs(r["lon"] - lon0) * 111319.9 * math.cos(dr(lat0))
        print(f"  offset from GPS: {dlat:.2f}m N-S, {dlon:.2f}m E-W, total {math.sqrt(dlat**2+dlon**2):.2f}m")

    # simulate pole detection: bottom-center-ish
    u2, v2 = cam["cx"], 3000
    print(f"\n--- Pole base ({u2:.0f}, {v2:.0f}) ---")
    r2 = pixel_to_geo(u2, v2, img, cam)
    if r2.get("error"):
        print(f"ERROR: {r2['error']}")
    else:
        print(f"  lat={r2['lat']:.8f} lon={r2['lon']:.8f} alt={r2['alt']:.3f}m t={r2.get('t',0):.1f}m")
        print(f"  ray dir ENU: {r2['ray_enu']}")
