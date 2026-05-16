# 水陆边界算法 Benchmark 设计

## 目标

在 1D 合成剖面上定量评估不同水陆边界检测算法的边界恢复精度，观察三项关键表现：

1. **边界偏移**：算法判定的水陆分界与 ground truth 的距离
2. **大面积水淹**：陆地段被判为水面（假阳性泛滥）
3. **碎片化水坑**：陆地段上大量孤立的高 lakeMask 段

对比多种梯度算子、范数、预平滑、多尺度策略和决策函数组合，选出 shader 可实现的最优方案。

## 当前算法回顾

**vertex shader** (`earth.vert.glsl:146-160`)：

```
lakeStep = uLakeRange / uTextureSize
lake_h_r/l/u/d = sampleFiltered(sampleUV ± lakeStep, mode=2.0)  # Catmull-Rom
terrainSteepness = sqrt(lake_du² + lake_dv²)
lakeMask = (elevation > 0.1)
  ? (1 - smoothstep(uLakeFlatness*0.3, uLakeFlatness, terrainSteepness))
  : 0.0
```

核心思想：高程梯度幅值 → smoothstep → 连续 lakeMask。

## 测试范围

### 测试维度：仅 1D u 轴剖面

1D 测试聚焦梯度算子和决策函数，2D 几何正确性靠现有 vitest 保证。截面沿 u 轴（东西方向）切取，此时简化为一维：`terrainSteepness = |lake_du|`（dv = 0）。

### 降采样测试：暂不测试

当前只测试 z=12 精度下的算法表现。多 zoom 级别的降采样+回插测试后续补充。

### 数据来源

| 类型 | 用途 | Ground Truth |
|------|------|-------------|
| JAXA 原始 GeoTIFF (~1m/px) | 统计参数提取 | OSM 水体边界 |
| 合成剖面（参数来自 JAXA 统计） | 定量评估 | 精确已知 |

## 合成剖面设计

### 模型

对于湖泊（单边界，一侧无限水域）：

```
h(x) =
  waterLevel                              for x < shoreline
  waterLevel + slope * (x - shoreline)
    + microRelief(x)                      for x >= shoreline
  + terrainRgbNoise(x)                    # terrian-rgb 0.1m 量化噪声
```

对于河流（双边界，两侧陆地）：

```
h(x) =
  landElev + slope*(x)                    for x < bank1
  waterLevel                              for bank1 <= x <= bank2
  landElev + slope*(x)                    for x > bank2
```

河流左右岸各算各的，共用算法。

### 参数

| 参数 | 含义 | 来源 |
|------|------|------|
| `waterLevel` | 水面高程 (m) | JAXA 水面段均值 |
| `slope` | 陆地坡度 (m/m) | JAXA 陆地段线性回归 |
| `microRelief` | 陆地微起伏 (σ) | JAXA 陆地段去趋势残差 std |
| `noise` | terrain-rgb 量化噪声 | σ=0m (理想) / 0.05m / 完整 encode-decode |

### 噪声模型

三种都测：

1. **σ = 0m** — 理想无噪声，看算法本底表现
2. **σ = 0.05m** — terrain-rgb 量化噪声理论值（0.1m step / sqrt(12) ≈ 0.029m，保守取 0.05m）
3. **完整编码链** — 合成剖面 → `encode(0.1m 量化)` → `decode` → 跑算法。精确模拟 terrain-rgb 真噪声

## 算法候选矩阵

所有算法均可 shader 实现。

### 维度 1：梯度算子 (G)

| ID | 名称 | 采样数/轴 | Shader texture fetches |
|----|------|----------|----------------------|
| G1 | Central difference | 2 | 4 |
| G2 | Forward difference | 1 | 2 |
| G3 | 3-point stencil | 3 | 6 |
| G4 | 5-point stencil | 5 | 10 |
| G5 | 3×3 Sobel | 3×3 | 8 |
| G6 | 3×3 Prewitt | 3×3 | 8 |

### 维度 2：陡峭度范数 (N)

| ID | 公式 |
|----|------|
| N1 | L2: sqrt(du² + dv²) |
| N2 | L1: |du| + |dv| |
| N3 | L∞: max(|du|, |dv|) |

### 维度 3：窗口统计量 (W)

替代梯度的方法，检测局部平坦度：

| ID | 方法 | 采样数 |
|----|------|--------|
| W1 | 3×3 variance σ² | 9 |
| W2 | 5×5 variance σ² | 25 |
| W3 | 3×3 range (max - min) | 9 |
| W4 | RMS roughness (local detrend) | 9+ |

### 维度 4：预平滑 (S)

| ID | 方法 |
|----|------|
| S0 | 无平滑（baseline） |
| S1 | 3×3 box blur 后梯度 |
| S2 | 3×3 Gaussian 后梯度 |

### 维度 5：多尺度 (M)

| ID | 方法 | Description |
|----|------|------------|
| M1 | 单尺度 range=1 | baseline |
| M2 | 大尺度 range>1 | 只检测大面积平坦 |
| M3 | 双尺度 min(grad_s, grad_l) | 都平坦才算湖 |
| M4 | 双尺度乘积 grad_s × grad_l | 抑制单尺度假阳性 |

### 维度 6：决策函数 (D)

| ID | 函数 | 说明 |
|----|------|------|
| D1 | smoothstep(0.3f, f, ts) | 当前 baseline |
| D2 | hard step(threshold) | 二值 |
| D3 | smoothstep(k*f, f, ts) | 可变 lo/hi 比率 (k = 0.1, 0.3, 0.5, 0.7) |

### 维度 7：Elevation guard (E)

| ID | 条件 |
|----|------|
| E1 | elevation > 0.1m（当前，硬编码） |
| E2 | elevation > -1m |
| E3 | 无 guard（纯梯度判定） |

## 评估策略

### 逐维度独立扫描

对每个维度，固定其他维度为 baseline（当前算法配置），逐个 sweep。第一轮完成后，用第一轮各维 best 更新 baseline，做第二轮。首轮独立报告（固定所有为初始 baseline），不交叉。

### Baseline 配置

| 维度 | Baseline |
|------|----------|
| G | G1 (central difference) |
| N | N1 (L2) |
| W | 不参与（用 G 替代） |
| S | S0 (无平滑) |
| M | M1 (range=1) |
| D | D1 (smoothstep 0.3f/f) |
| E | E3 (无 elevation guard) |

### 验证

用 sin 曲面验证 Python numpy 实现与 GLSL 的等价性——解析梯度已知，无插值歧义。

## 评估指标

三项实用指标：

**a) 边界偏移**：lakeMask 首次穿过 0.5 的位置与 ground truth 边界的距离（单位：米）。可接受范围：±2 个原始 texel（~60m @ z=12）。

**b) 大面积水淹**：从 ground truth 边界向陆侧延伸 N 个采样点，统计 lakeMask > 0.5 的面积占比。若 > 20% 的陆地侧窗口被判为水 → 标记为过度水淹。

**c) 碎片化水坑**：统计陆地段中 lakeMask > 0.5 的连通块数。若 ≥ 3 块间距 > 1 texel → 标记为碎片化。

## 数据 Pipeline

```
Phase 1: 数据准备
  1a. 查 OSM 水体边界（Overpass API），存 SQLite
  1b. 从 JAXA GeoTIFF 提取多水体 1D u 轴剖面
  1c. 标注真实水面段边界（OSM 矢量与剖面的交点）
  1d. 从 JAXA 剖面提取统计参数（slope, microRelief σ, etc.）
  1e. 基于统计参数生成合成剖面（ground truth 精确已知）

Phase 2: 算法实现
  2a. numpy 手写 GLSL 等价实现（1D 版本）
  2b. sin 曲面验证等价性

Phase 3: 算法评估
  3a. 全矩阵跑所有组合，真实 + 合成剖面
  3b. 算三项指标
  3c. 逐维度独立扫描

Phase 4: 汇总
  4a. 汇总表
  4b. 散点图（误判面积 vs 边界偏移）
  4c. 典型失败案例剖面图
```

## SQLite Schema

```sql
-- 水体/目标
CREATE TABLE water_bodies (
  id INTEGER PRIMARY KEY,
  name TEXT,           -- '西湖', '长江-南京段'
  type TEXT,           -- 'lake', 'river'
  osm_id INTEGER,      -- OSM relation/way ID
  center_lat REAL,
  center_lon REAL
);

-- 剖面线
CREATE TABLE profiles (
  id INTEGER PRIMARY KEY,
  water_body_id INTEGER REFERENCES water_bodies(id),
  label TEXT,          -- 'westlake_u_01'
  start_lat REAL, start_lon REAL,
  end_lat REAL, end_lon REAL,
  direction TEXT,      -- 'u', 'v', 'normal'
  sample_count INTEGER
);

-- 高程序列
CREATE TABLE elevation_samples (
  profile_id INTEGER REFERENCES profiles(id),
  idx INTEGER,
  dist_m REAL,          -- 沿剖面距离（m）
  lon REAL, lat REAL,
  elev_raw REAL,        -- JAXA 原始高程（m）
  elev_terrain_rgb REAL,-- terrain-rgb 量化后（0.1m step）
  PRIMARY KEY (profile_id, idx)
);

-- ground truth 边界
CREATE TABLE shore_labels (
  profile_id INTEGER REFERENCES profiles(id),
  idx_start INTEGER,    -- 水面段起始采样点
  idx_end INTEGER,      -- 水面段结束采样点
  source TEXT,          -- 'osm' or 'manual'
  confidence REAL       -- 0-1
);

-- 算法配置（每个被测组合一行）
CREATE TABLE algorithm_configs (
  id INTEGER PRIMARY KEY,
  label TEXT,            -- 'G1_N1_S0_M1_D1_E3'
  grad_op TEXT,          -- G1..G6
  norm TEXT,             -- N1..N3
  window_stat TEXT,      -- NULL or W1..W4
  pre_smooth TEXT,       -- S0..S2
  multi_scale TEXT,      -- M1..M4
  decision_fn TEXT,      -- D1..D3
  elev_guard TEXT,       -- E1..E3
  extra_params TEXT      -- JSON, e.g. {"range":4,"k":0.3} for M2/D3 variants
);

-- 合成剖面生成参数（可复现）
CREATE TABLE synthetic_params (
  profile_id INTEGER PRIMARY KEY REFERENCES profiles(id),
  water_level REAL,      -- 水面高程 (m)
  slope REAL,            -- 陆地坡度 (m/m)
  micro_relief_sigma REAL, -- 微起伏 σ (m)
  source_profile_id INTEGER REFERENCES profiles(id)  -- 统计来源的 JAXA 剖面
);

-- Benchmark 运行记录
CREATE TABLE benchmark_runs (
  id INTEGER PRIMARY KEY,
  algorithm_id INTEGER REFERENCES algorithm_configs(id),
  profile_id INTEGER REFERENCES profiles(id),
  noise_model TEXT,      -- 'ideal', 'white_005', 'terrain_rgb_chain'
  run_at TEXT            -- timestamp
);

-- 三项评估指标
CREATE TABLE benchmark_metrics (
  run_id INTEGER PRIMARY KEY REFERENCES benchmark_runs(id),
  boundary_offset_m REAL,       -- 边界偏移（m），正=向陆偏，负=向水偏
  flood_ratio REAL,             -- 陆侧窗口 lakeMask>0.5 的比例
  fragment_count INTEGER,       -- 陆侧 lakeMask>0.5 的连通块数
  lake_mask_rmse REAL,          -- vs ideal step function 的 RMSE
  transition_width_m REAL       -- smoothstep 过渡带宽度（m）
);

-- 逐采样点 lakeMask 明细（用于绘对比图）
CREATE TABLE lake_mask_detail (
  run_id INTEGER REFERENCES benchmark_runs(id),
  idx INTEGER,           -- 剖面采样序号
  dist_m REAL,           -- 沿剖面距离
  elevation REAL,        -- 输入高程
  lake_mask REAL,        -- 算法输出 [0,1]
  PRIMARY KEY (run_id, idx)
);

-- 维度扫描结果（汇总）
CREATE TABLE dim_sweep_results (
  id INTEGER PRIMARY KEY,
  dim_name TEXT,         -- 'grad_op', 'norm', 'pre_smooth', 'multi_scale', 'decision', 'elev_guard'
  variant_id TEXT,       -- G1, G2, ...
  avg_offset_m REAL,     -- 所有剖面平均边界偏移
  avg_flood_ratio REAL,  -- 所有剖面平均水淹比例
  avg_fragment_count REAL,
  rank_offset INTEGER,   -- 在该维度中的排名（偏移越小越好）
  rank_flood INTEGER     -- 水淹排名
);
```

## 脚本结构

```
scripts/
├── sample_profiles.py      # 数据准备：OSM 查询 + JAXA 剖面提取 → SQLite
├── generate_synthetic.py   # 合成剖面生成（参数来自 JAXA 统计）→ SQLite
├── lake_algorithms.py       # numpy GLSL 等价算法库（纯函数）
├── benchmark_lake.py        # 算法评估 + 指标计算 → SQLite
├── plot_benchmark.py        # 从 SQLite 读结果出图 → charts/
```

## 输入

- `/home/dev/jaxa_storage/` — JAXA 原始 DEM GeoTIFF（1°×1°，~1m/px），只读
- Overpass API — OSM 水体边界矢量，在线查询

## 输出

### SQLite 数据库

位于 `tmp/lake_benchmark/lake_boundary.db`（已 gitignore）。

| 表 | 内容 |
|----|------|
| `water_bodies` | 被测水体元数据 |
| `profiles` | 剖面线定义 |
| `elevation_samples` | 1D 高程采样点 |
| `shore_labels` | ground truth 水陆边界 |
| `synthetic_params` | 合成剖面生成参数（可复现） |
| `algorithm_configs` | 被测算法组合定义 |
| `benchmark_runs` | 每次运行记录 |
| `benchmark_metrics` | 三项指标 + RMSE + 过渡带宽度 |
| `lake_mask_detail` | 逐采样点 lakeMask（用于绘对比图） |
| `dim_sweep_results` | 逐维度扫描汇总 |

### 图表

位于 `tmp/lake_benchmark/charts/`（已 gitignore）。

```
tmp/lake_benchmark/charts/
├── scatter_overview.png    # 概览散点图（误判面积 vs 边界偏移）
├── dim_gradient.png        # 梯度算子对比
├── dim_norm.png            # 范数对比
├── dim_smooth.png          # 预平滑对比
├── dim_multiscale.png      # 多尺度对比
├── dim_decision.png        # 决策函数对比
└── profile_*.png           # 各剖面算法叠加对比图
```

## 待补充

- 降采样链测试（z=12 → z=6，不同插值方式）
- 2D 合成地形块测试
- 河流多段剖面数据采集
