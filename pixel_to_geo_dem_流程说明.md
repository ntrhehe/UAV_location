# pixel_to_geo_dem.py 计算流程说明

## 整体目标

输入：DJI 照片的**像素坐标** `(u, v)` + **DSM 高程模型**
输出：该像素对应地面点的 **CGCS2000 大地坐标 (EPSG:4490)** `(lon, lat, alt)`

---

## 流程全景图

```
像素 (u, v)
  │
  ▼
Step 1: 解析 XMP 元数据 ──→ 相机位姿 (lat0, lon0, alt0, yaw, pitch, roll) + 内参
  │
  ▼
Step 2: 像素归一化（可选畸变校正）──→ 归一化相机坐标 (xn, yn)
  │
  ▼
Step 3: 相机方向向量 → ENU 方向向量 ──→ v_enu（单位向量，从相机指向地面点）
  │              ENU 原点 = 相机位置
  │
  ▼
Step 4: ENU 方向 → ECEF 方向（旋转矩阵）
  │        + 相机原点 WGS84 → ECEF
  │
  ▼
Step 5: 在 ECEF 空间中沿射线步进，与 DSM 求交
  │        每步: ECEF → WGS84 → 查 DSM 高程
  │        穿过 DSM 表面时做二分法精化
  │
  ▼
Step 6: 返回 CGCS2000 坐标 (lon, lat, alt)
```

---

## Step 1：解析 XMP 元数据

**函数：`parse_dji_xmp(jpg_path)`**

从 DJI JPG 的 XMP 标签中提取相机参数：

| 参数 | XMP 标签 | 说明 |
|------|----------|------|
| `lat, lon` | `GpsLatitude`, `GpsLongitude` | 相机 GPS 经纬度 |
| `abs_alt` | `AbsoluteAltitude` | 绝对海拔（WGS84 椭球高） |
| `rel_alt` | `RelativeAltitude` | 相对起飞点高度 |
| `pitch` | `GimbalPitchDegree` | 云台俯仰角，0=水平，-90=正下视 |
| `roll` | `GimbalRollDegree` | 云台横滚角 |
| `flight_yaw` | `FlightYawDegree` | 飞机航向角（机头朝向） |
| `gimbal_yaw` | `GimbalYawDegree` | 云台相对机头的偏航角 |
| `effective_yaw` | **合成值** | 有效偏航角（见下方） |
| `fx, fy` | `CalibratedFocalLength` | 焦距（像素单位） |
| `cx, cy` | `CalibratedOpticalCenterX/Y` | 主点坐标 |

**effective_yaw 合成逻辑：**
- 正下视（pitch 接近 -90°，容差 0.5°）：`effective_yaw = FlightYaw`
- 倾斜摄影（其他情况）：`effective_yaw = FlightYaw + GimbalYaw`

---

## Step 2：像素归一化

**函数：`pixel_to_cgcs2000()` 内部**

### 情况 A：有内参文件 cam.txt → 畸变校正

用 `undistort()` 函数做 Brown-Conrady 模型校正（8 次固定点迭代）：

```
xd = (u - cx) / fx      # 归一化畸变坐标
yd = (v - cy) / fy

迭代 8 次:
    r² = xu² + yu²
    radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
    tx = 2·p1·xu·yu + p2·(r² + 2·xu²)
    ty = p1·(r² + 2·yu²) + 2·p2·xu·yu
    xu = (xd - tx) / radial
    yu = (yd - ty) / radial
```

### 情况 B：无内参文件 → 直接用 XMP 内参

```
xn = (u - cx) / fx
yn = (v - cy) / fy
```

### 构造相机方向向量

```python
v_cam = np.array([xn, yn, 1.0])       # (右, 下, 光轴方向)
v_cam /= np.linalg.norm(v_cam)        # 归一化为单位向量
```

相机坐标系：
- X 轴：图像向右
- Y 轴：图像向下
- Z 轴：光轴向前（镜头朝向）

---

## Step 3：相机坐标系 → ENU 方向向量

**函数：`make_rotation_dji(yaw, pitch, roll)`**

ENU 坐标系定义：
- E 轴：指向正东
- N 轴：指向正北
- U 轴：指向天顶

**ENU 原点 = 相机 GPS 位置 `(lat0, lon0, alt0)`**

### 旋转矩阵构造

```
1. 由 yaw、pitch 计算 forward 向量（相机光轴在 ENU 中的方向）：
   fwd = (sin(yaw)·cos(pitch), cos(yaw)·cos(pitch), sin(pitch))

2. 计算 right 向量（水平向右，与 forward 垂直）：
   right0 = normalize(-fwd_y, fwd_x, 0)

3. down = forward × right（叉积）

4. 绕 forward 轴旋转 roll 角：
   right = cos(roll)·right0 + sin(roll)·down0
   down  = -sin(roll)·right0 + cos(roll)·down0

5. R = [right, down, forward]ᵀ  (3×3 矩阵，列排列)
```

### 应用旋转

```python
v_enu = R @ v_cam        # v_cam = [xn, yn, 1.0]（归一化相机方向）
v_enu /= np.linalg.norm(v_enu)  # 确保是单位向量
```

**此时 `v_enu` 的含义：**

从相机位置（ENU 原点）出发、指向地面像素点的**单位方向向量**，在 ENU 坐标系中的分量为 `(east, north, up)`。

举例：`v_enu = (0.3, -0.2, -0.93)` 表示
- 向东 0.3 米
- 向南 0.2 米
- 向下 0.93 米
（每前进 1 米距离的变化量）

射线方程：
```
地面点(ENU) = ENU原点(相机位置) + 距离 × v_enu
```

---

## Step 4：ENU → ECEF 转换

**函数：`ray_dem_intersection()` 前段**

### 4a. 相机原点转 ECEF

```python
origin = wgs84_to_ecef(lat0, lon0, alt0)
```

```
lat, lon = deg2rad(lat0), deg2rad(lon0)
sin_lat = sin(lat)
N = a / sqrt(1 - e²·sin²(lat))

X = (N + alt) · cos(lat) · cos(lon)
Y = (N + alt) · cos(lat) · sin(lon)
Z = (N·(1 - e²) + alt) · sin(lat)
```

### 4b. ENU 方向转 ECEF 方向

```python
dir_ecef_x = -slon·e - slat·clon·n + clat·clon·u
dir_ecef_y =  clon·e - slat·slon·n + clat·slon·u
dir_ecef_z =           clat·n     +  slat·u
```

对应的旋转矩阵：
```
        [ -sin(lon)    cos(lon)       0       ]
R =     [ -sin(lat)·cos(lon)  -sin(lat)·sin(lon)   cos(lat) ]
        [  cos(lat)·cos(lon)   cos(lat)·sin(lon)   sin(lat) ]
```

**为什么转到 ECEF？**
- 射线在 ECEF 中是**直线**，步进简单
- 在 ENU 中走远距离会因地球曲率引入误差
- ECEF 步进 + 每步 `ecef_to_wgs84` 转回地理坐标，自动考虑曲率

---

## Step 5：射线与 DSM 求交

**函数：`ray_dem_intersection()`**

### 阶段 I：粗步进

```
t = 0
while t < 8000 米:
    t += 3.0 米
    pt = origin + t × dir_ecef
    lat_pt, lon_pt, alt_pt = ecef_to_wgs84(pt)
    dem_h = dsm.query_elevation(lat_pt, lon_pt)
    if alt_pt < dem_h:    ← 射线进入地面以下
        进入二分精化
```

**`ecef_to_wgs84()` 反算公式：**
```
b = a·sqrt(1 - e²)
ep² = (a² - b²) / b²

lon = atan2(Y, X)
p = sqrt(X² + Y²)
θ = atan2(Z·a, p·b)
lat = atan2(Z + ep²·b·sin³(θ), p - e²·a·cos³(θ))

N = a / sqrt(1 - e²·sin²(lat))
alt = p / cos(lat) - N
```

### 阶段 II：二分法精化（16 次迭代）

```
lo = t - step, hi = t
迭代 16 次:
    tm = (lo + hi) / 2
    pm = origin + tm × dir_ecef
    lat_m, lon_m, alt_m = ecef_to_wgs84(pm)
    dem_m = dsm.query_elevation(lat_m, lon_m)
    if alt_m < dem_m:   hi = tm    # 在地下 → 缩上界
    else:               lo = tm    # 在地上 → 抬下界

t_final = (lo + hi) / 2
```

16 次二分的精度：`3.0 / 2¹⁶ ≈ 4.6e-5 米`

### DSM 查询细节

**`DSMQuery.query_elevation(lat, lon)`：**

1. **坐标转换**：`EPSG:4490 → DSM 投影 CRS`（如 EPSG:4547），再通过仿射变换转像素行列号
2. **读取 2×2 窗口**做双线性插值
3. **NoData 处理**：不足 2 个有效值时用最近邻，否则用均值填充

双线性插值公式：
```
top = v00·(1-fx) + v10·fx
bot = v01·(1-fx) + v11·fx
elev = top·(1-fy) + bot·fy
```

---

## Step 6：返回结果

```python
return {
    'lon': result['lon'],        # CGCS2000 经度
    'lat': result['lat'],        # CGCS2000 纬度
    'alt': result['alt'],        # WGS84 椭球高（与 CGCS2000 一致）
    'dist_m': result['dist'],    # 相机到地面点空间距离（米）
}
```

---

## 坐标系变换关系总图

```
                    make_rotation_dji()                        由经纬度构造
  像素 (u, v) ───→ 相机坐标系 ────────────→  ENU 坐标系 ────────────→  ECEF 坐标系
                   (right,down,forward)    (east,north,up)          (X, Y, Z)
                                                ↑                        │
                                                │ 原点 = 相机位置          │ ecef_to_wgs84()
                                                │ (lat0,lon0,alt0)       ▼
                                                                   CGCS2000 大地坐标
                                                                   (lon, lat, alt)
```

---

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `a` | 6378137.0 | 椭球长半轴（WGS84 / CGCS2000） |
| `f` | 1/298.257222101 | 椭球扁率 |
| `e²` | 2f - f² | 第一偏心率平方 |
| 步长 | 3.0 米 | ECEF 射线粗步进 |
| 最大距离 | 8000 米 | 射线搜索上限 |
| 二分次数 | 16 | 精化迭代次数 |
| 下视容差 | 0.5° | 判断 pitch 是否接近 -90° |
