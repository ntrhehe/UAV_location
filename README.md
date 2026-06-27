# UAV Geo-Localization

无人机航拍图像的**像素 → 地理坐标**映射工具集。输入 DJI 航拍 JPG（含 XMP 元数据）和 DSM 高程模型，输出 CGCS2000 (EPSG:4490) 大地坐标。

## 项目用途

从无人机倾斜/正射照片中，精确计算任意像素对应的地面经纬度坐标，支持：

* 电杆、路牌等目标的像素坐标 → 地理坐标定位
* 单张照片的地面覆盖范围（footprint）计算
* 航拍照片批量正射校正（生成 GeoTIFF）
* 照片空间覆盖筛选（用最少照片覆盖指定区域）

## 总体思路

`JPG (含XMP位姿) ──→ 相机光线方向向量 ──→ ENU 坐标系 │ ▼ ECEF 空间射线步进 │ ▼ ┌──── 与 DSM 求交 ────┐ │ │ 有 DSM 无 DSM │ │ │ 精确地面交点 平坦地面假设 │ │ │ ▼ ▼ CGCS2000 坐标 (粗略估计) (lon, lat, alt)`

核心方法：从相机位置出发，沿像素对应的光线方向在 ECEF 空间中步进，每步查询 DSM 高程，找到射线与地表的精确交点。

## 脚本说明

### src/pixel_to_geo_dem.py — 核心定位引擎

单张照片 **像素 → CGCS2000 坐标** 的核心模块。流程：

1. 解析 DJI JPG 的 XMP 元数据（GPS 位置、云台姿态、相机内参）
2. 像素归一化 + 可选 Brown-Conrady 畸变校正
3. 构造相机→ENU 旋转矩阵（yaw-pitch-roll）
4. ENU→ECEF 转换，在 ECEF 空间做射线步进
5. 每步查询 DSM 高程，穿越地表时二分法精化（精度 ~0.05mm）

**用法：**
python  src/pixel_to_geo_dem.py <照片.JPG> <DSM.tif> <像素u> <像素v>

### src/geo_reference.py — 批量正射校正

将 DJI 照片批量生成带地理坐标的正射 GeoTIFF。

* 正下视照片：用 8 个 GCP（四角+四边中点）做仿射校正
* 倾斜照片：用 GCP 网格（可设步长）做 Thin Plate Spline 扭曲校正
* 支持外部 CSV 优化位姿和相机内参文件

**用法：**
python  src/geo_reference.py --img <照片目录> --dsm <DSM.tif> --out <输出目录> [--cam cam.txt] [--csv pos.csv]

### src/image_footprint.py — 地面覆盖范围计算

计算单张照片在地面的覆盖范围（四角 + 边中点 + 中心共 9 个采样点），输出每个点的经纬度和边界框。

**用法：**
python src/image_footprint.py <照片.JPG> <DSM.tif> [--cam cam.txt] [--csv pos.csv]

### src/full_coverage_selector.py — 空间覆盖筛选

从大量航拍照片中自动选出**最少照片**覆盖指定 ROI 区域。

* 支持 SHP 多边形或 DOM 边界作为 ROI
* DSM 精确计算每张照片的地面 footprint
* GPS 粗过滤 + 贪心网格覆盖算法
* 输出：选中照片列表、GeoJSON 可视化、覆盖图

**用法：**
python src/full_coverage_selector.py --img <照片目录> --dsm <DSM.tif> --shp <ROI.shp> --out <输出目录>

### src/gen_all_footprints_shp.py — 航向抽稀选图

航拍照片航向重叠率高达 70%，通过在航向上按文件名编号等间隔取图即可大幅减少冗余。所有航线全保留以保证旁向边缘全覆盖。后续可配合裁剪边缘 10% 减轻变形对配准的影响。

**功能：**
- 按文件名末尾编号排序 → 每隔 N 张取 1 张（--step 参数控制）
- DSM 精确计算 footprint + ROI 边界过滤
- 输出选中照片的 footprint SHP + 复制 JPG 到指定目录

**用法：**
python src/gen_all_footprints_shp.py <JPG_DIR> <DSM_TIF> <ROI_SHP> <OUT_SHP> [--step 5] [--copy_dir <DIR>]

### src/pixel_to_geo.py — 简易版定位（无 DSM）

pixel_to_geo_dem.py 的简化版。假设平坦地面（相机相对高度作为地面高程），不用 DSM。适合快速估算或无可用地形数据时使用。

## 文件说明

|文件|说明|
|-|-|
|cam.txt|相机内参（焦距、主点、畸变系数 Brown-Conrady）|
|ilter.md|项目背景与技术路线文档|
|geo_reference_原理说明.md|地理校正原理：透视 vs 仿射的数学解释|
|pixel_to_geo_dem_流程说明.md|像素→地理坐标的完整计算流程与公式|
|精度不够的原因.txt|单张照片定位精度不足的原因分析|

## 依赖

`rasterio  numpy  Pillow  pyproj fiona`

## 坐标系

* 输入相机位姿：WGS84 经纬高（DJI XMP 标准）
* 输出地理坐标：**CGCS2000 (EPSG:4490)**，与 WGS84 在厘米级一致
* 高程基准：WGS84 椭球高（CGCS2000 椭球一致）
* DSM 投影：支持任意投影 CRS，自动转换

## 常用流程

* 使用gen_all_footprints_shp.py筛选图片
* 使用geo_reference.py进行地理投影
