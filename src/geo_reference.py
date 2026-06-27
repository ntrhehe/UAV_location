# -*- coding: utf-8 -*-
import sys, numpy as np, math, os, csv as _csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pixel_to_geo_dem import (
    load_camera_intrinsics, parse_dji_xmp, pixel_to_cgcs2000, DSMQuery,
)

NADIR_TOL = 0.5
def is_nadir(pitch): return abs(pitch + 90.0) < NADIR_TOL

def parse_csv_poses(csv_path):
    csv_path = Path(csv_path)
    poses = {}
    with open(str(csv_path), encoding='gbk', errors='ignore') as f:
        reader = _csv.reader(f)
        for row in reader:
            if not row or len(row) < 7: continue
            fp = row[0].strip()
            if '.JPG' not in fp.upper(): continue
            name = Path(fp).name
            poses[name] = dict(name=name, lat=float(row[1]), lon=float(row[2]),
                               alt=float(row[3]), yaw=float(row[4]),
                               pitch=float(row[5]), roll=float(row[6]))
    return poses

def generate_gcp_grid(pose, dsm, intrinsics, img_w, img_h, grid_step):
    rows = list(range(0, img_h, grid_step))
    cols = list(range(0, img_w, grid_step))
    if not rows or rows[-1] != img_h - 1: rows.append(img_h - 1)
    if not cols or cols[-1] != img_w - 1: cols.append(img_w - 1)
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
        (0,0,'tl'),(img_w-1,0,'tr'),(img_w-1,img_h-1,'br'),(0,img_h-1,'bl'),
        (img_w//2,0,'tc'),(img_w-1,img_h//2,'rc'),(img_w//2,img_h-1,'bc'),(0,img_h//2,'lc'),
    ]
    from rasterio.control import GroundControlPoint
    gcps = []
    for u,v,label in samples:
        res = pixel_to_cgcs2000(u, v, pose, dsm, intrinsics)
        if 'error' not in res:
            gcps.append(GroundControlPoint(row=v, col=u, x=res['lon'], y=res['lat']))
    return gcps

def _crop_and_offset_gcps(gcps_raw, x0, y0, wc, hc):
    from rasterio.control import GroundControlPoint
    gcps = []
    for gcp in gcps_raw:
        nc, nr = gcp.col - x0, gcp.row - y0
        if 0 <= nc < wc and 0 <= nr < hc:
            gcps.append(GroundControlPoint(row=nr, col=nc, x=gcp.x, y=gcp.y))
    return gcps

def _write_nadir_affine(jpg_path, pose, dsm, intrinsics, output_path, crs, crop_margin):
    import rasterio
    from rasterio.transform import from_gcps
    from rasterio.warp import reproject, Resampling
    from PIL import Image
    img = Image.open(str(jpg_path))
    arr = np.array(img)
    h, w = arr.shape[:2]
    bands = arr.shape[2] if arr.ndim == 3 else 1
    dtype = arr.dtype
    if crop_margin > 0:
        x0=int(w*crop_margin); x1=int(w*(1.0-crop_margin))
        y0=int(h*crop_margin); y1=int(h*(1.0-crop_margin))
        arr = arr[y0:y1,x0:x1] if arr.ndim==2 else arr[y0:y1,x0:x1,:]
        hc,wc = arr.shape[:2]
    else:
        x0,y0,hc,wc = 0,0,h,w

    # 对裁剪后的区域重新生成GCP：4角 + 4边中点
    from rasterio.control import GroundControlPoint
    gcps = []
    samples = [
        (0,0),(wc-1,0),(wc-1,hc-1),(0,hc-1),
        (wc//2,0),(wc-1,hc//2),(wc//2,hc-1),(0,hc//2),
    ]
    for u,v in samples:
        # 映射回原图坐标
        u_orig, v_orig = u + x0, v + y0
        res = pixel_to_cgcs2000(u_orig, v_orig, pose, dsm, intrinsics)
        if 'error' not in res:
            gcps.append(GroundControlPoint(row=v, col=u, x=res['lon'], y=res['lat']))

    if len(gcps) < 4:
        print('  FAILED (only %d GCPs after crop)' % len(gcps))
        return False
    print('  nadir->affine (%d GCPs, crop=%.2f)' % (len(gcps), crop_margin), end='', flush=True)
    aff = from_gcps(gcps)
    if aff is None:
        print('  FAILED (from_gcps)')
        return False
    lons,lats=[],[]
    for c,r in [(0,0),(wc,0),(wc,hc),(0,hc)]:
        lo,la = aff*(c,r); lons.append(lo); lats.append(la)
    lo_min,lo_max = min(lons),max(lons)
    la_min,la_max = min(lats),max(lats)
    gsd_lo,gsd_la = abs(aff[0]),abs(aff[4])
    if gsd_lo<=0 or gsd_la<=0:
        print('  FAILED (zero GSD)')
        return False
    ow=max(1,int(np.ceil((lo_max-lo_min)/gsd_lo)))
    oh=max(1,int(np.ceil((la_max-la_min)/gsd_la)))
    ot = rasterio.transform.from_bounds(lo_min,la_min,lo_max,la_max,ow,oh)
    prof = dict(driver='GTiff',height=oh,width=ow,count=bands,dtype=dtype,crs=crs,transform=ot,compress='lzw')
    with rasterio.open(str(output_path),'w',**prof) as dst:
        for i in range(bands):
            sb = arr[:,:,i] if bands>1 else arr
            db = np.empty((oh,ow),dtype=dtype)
            reproject(source=sb,destination=db,src_transform=aff,src_crs=crs,
                      dst_transform=ot,dst_crs=crs,resampling=Resampling.bilinear)
            dst.write(db,i+1)
    print('  OK')
    return True


def _write_oblique_warp(jpg_path, pose, dsm, intrinsics, output_path, crs, grid_step, crop_margin):
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform
    from rasterio.io import MemoryFile
    from PIL import Image
    img = Image.open(str(jpg_path))
    arr = np.array(img)
    h,w = arr.shape[:2]
    bands = arr.shape[2] if arr.ndim==3 else 1
    dtype = arr.dtype
    if crop_margin > 0:
        x0=int(w*crop_margin); x1=int(w*(1.0-crop_margin))
        y0=int(h*crop_margin); y1=int(h*(1.0-crop_margin))
        arr = arr[y0:y1,x0:x1] if arr.ndim==2 else arr[y0:y1,x0:x1,:]
        hc,wc = arr.shape[:2]
    else:
        x0,y0,hc,wc = 0,0,h,w
    gcps_raw = generate_gcp_grid(pose, dsm, intrinsics, w, h, grid_step)
    gcps = _crop_and_offset_gcps(gcps_raw, x0, y0, wc, hc)
    if len(gcps) < 4:
        print('  FAILED (only %d GCPs after crop)' % len(gcps))
        return False
    nc=(w-1)//grid_step+2; nr=(h-1)//grid_step+2
    print('  oblique GCP: %dx%d -> %d valid, crop=%.2f' % (nc,nr,len(gcps),crop_margin), end='', flush=True)
    st = rasterio.transform.Affine(1.0,0.0,0.0,0.0,1.0,0.0)
    with MemoryFile() as mf:
        with mf.open(driver='GTiff',height=hc,width=wc,count=bands,dtype=dtype,crs=crs,transform=st,gcps=gcps) as mem:
            for i in range(bands):
                mem.write(arr[:,:,i] if bands>1 else arr, i+1)
        with mf.open() as src:
            dt,dw,dh = calculate_default_transform(
                src.crs,crs,src.width,src.height,
                left=src.bounds.left,bottom=src.bounds.bottom,
                right=src.bounds.right,top=src.bounds.top,gcps=src.gcps)
            dp = dict(driver='GTiff',height=dh,width=dw,count=bands,dtype=dtype,crs=crs,transform=dt,compress='lzw')
            with rasterio.open(str(output_path),'w',**dp) as dst:
                for i in range(1,bands+1):
                    reproject(source=rasterio.band(src,i),destination=rasterio.band(dst,i),
                              src_transform=src.transform,src_crs=src.crs,
                              dst_transform=dt,dst_crs=crs,src_gcps=src.gcps,
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
    ap = argparse.ArgumentParser(description='Georeference UAV photos to GeoTIFF')
    ap.add_argument('--img', required=True)
    ap.add_argument('--dsm', required=True)
    ap.add_argument('--out', default='./geotiff_output')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--cam', default=None)
    ap.add_argument('--crs', default='EPSG:4490')
    ap.add_argument('--grid-step', type=int, default=500)
    ap.add_argument('--crop-margin', type=float, default=0.10)
    a = ap.parse_args()
    sd = Path(__file__).parent
    img_dir = Path(a.img).resolve()
    dsm_path = Path(a.dsm).resolve()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print('  img        : %s' % img_dir)
    print('  DSM        : %s' % dsm_path)
    print('  out        : %s' % out_dir)
    print('  crop-margin: %.2f' % a.crop_margin)
    print()
    intrinsics = None
    if a.cam:
        intrinsics = load_camera_intrinsics(str(Path(a.cam).resolve()))
        print('[1/4] Intrinsics loaded: F=%.1f' % intrinsics['fx'])
    else:
        print('[1/4] Intrinsics: from XMP')
    all_poses = None
    if a.csv:
        all_poses = parse_csv_poses(str(Path(a.csv).resolve()))
        print('[2/4] CSV: %d poses' % len(all_poses))
    else:
        print('[2/4] Poses: from XMP')
    selected = sorted(p.name for p in img_dir.iterdir()
                      if p.suffix.upper() in ('.JPG','.JPEG','.PNG','.TIF','.TIFF'))
    print('  %d images' % len(selected))
    if not selected: return
    print('[ ] Georeferencing ...')
    dsm = DSMQuery(str(dsm_path))
    success, failed = 0, []
    for idx, name in enumerate(selected, 1):
        jp = img_dir / name
        if not jp.exists():
            failed.append((name, 'not found'))
            continue
        if all_poses:
            rec = all_poses.get(name)
            if rec is None:
                failed.append((name, 'no CSV pose'))
                continue
            try:
                pose = parse_dji_xmp(str(jp))
            except Exception as e:
                failed.append((name, 'XMP parse error: %s' % e))
                continue
            pose['lat']=rec['lat']; pose['lon']=rec['lon']
            pose['abs_alt']=rec['alt']; pose['roll']=0.0
        else:
            try:
                pose = parse_dji_xmp(str(jp)); pose['roll']=0.0
            except Exception as e:
                failed.append((name, 'XMP parse error: %s' % e))
                continue
        print('  [%d/%d] %s' % (idx, len(selected), name), end='', flush=True)
        op = out_dir / (Path(name).stem + '.tif')
        ok = write_geotiff(str(jp), pose, dsm, intrinsics, str(op),
                           crs=a.crs, grid_step=a.grid_step, crop_margin=a.crop_margin)
        if ok:
            success += 1
            print('\r  [%d/%d] %s  -> %s' % (idx, len(selected), name, op.name))
        else:
            print('\r  [%d/%d] %s  FAILED' % (idx, len(selected), name))
            failed.append((name, 'warp failed'))
    print('='*52)
    print('  Success: %d/%d' % (success, len(selected)))
    if failed:
        print('  Failed:')
        for n,r in failed: print('    - %s (%s)' % (n,r))
    print('  Output: %s' % out_dir)
    print('='*52)

if __name__ == '__main__':
    main()
