# 农险气象格点数据处理链路 (agromet)

面向农业保险理赔的气象格点数据处理平台：把多源、异构的 NetCDF / GRIB
预报与再分析场，统一到同一套经纬网格，做面积加权的区县统计、可配置的
旱涝冻灾害判定，并支持理赔点位的格点级溯源。提供 FastAPI + 原生前端网页。

## 需求对应

| # | 需求 | 实现位置 |
|---|------|----------|
| 1 | NetCDF/GRIB 读取，属性不规范也认得 | `agromet/io.py`, `agromet/units.py` |
| 2 | 多网格统一插值，方法可选 | `agromet/regrid.py`（nearest/linear/cubic/conservative）|
| 3 | 区县面积加权分区统计，跨格点按占比 | `agromet/zones.py` |
| 4 | 逐小时→逐日，降水累加/温度极值不混 | `agromet/resample.py` |
| 5 | 灾害指标公式可配 | `config/indicators.yaml`, `agromet/indicators.py` |
| 6 | 理赔判定可追溯到格点与原始值 | `agromet/lineage.py`, `agromet/regrid.py` |
| 7 | 全年分块处理，内存 ≤ 4 GiB | `agromet/pipeline.py`（逐月分块 + `MemoryGuard`）|

## 安装

```powershell
py -m pip install -r requirements.txt
```

GRIB 支持依赖 `cfgrib` + `eccodes`（pip wheel 自带 ecCodes 二进制库，
Windows 上无需单独安装）。

## 一键运行

```powershell
# 1) 生成一年（测试默认裁剪到 1、3 月）三套异构合成源，内含已知旱情/霜冻
py -I scripts/run_pipeline.py synth

# 2) 逐月分块跑完整链路（插值方法可选 nearest/linear/cubic/conservative）
py -I scripts/run_pipeline.py run --method linear

# 或一步到位
py -I scripts/run_pipeline.py all --method linear

# 3) 启动网页（默认 http://127.0.0.1:8000）
py -I scripts/run_pipeline.py web
```

## 网页五个板块

1. **数据集**：扫描 `data/raw`，展示格式、网格、变量、时间范围，以及每个变量
   的原始名→标准变量、原始单位→标准单位的识别审计。
2. **处理任务**：选数据集和插值方法发起处理，展示分块数、峰值内存与逐阶段
   内存轨迹（4 GiB 守卫）。
3. **区县统计**：区县地图着色、时间序列（温度面积加权平均、降水平均强度/总量）。
4. **灾害判定**：指标公式表、命中网格图、区县级灾害事件清单。
5. **理赔查询**：输入经纬度与日期段，生成溯源报告——理赔点落入的目标格点、
   每个数据源参与插值的源格点坐标/权重、这些格点的小时级原始值，可用
   `Σ(源值 × 权重)` 严格复算目标值。

前端为单文件原生 HTML/CSS/JS（`agromet/web/static/index.html`），地图用
Canvas 等距投影手绘、图表用内联 SVG，无外部 CDN，可在内网离线运行。

## 合成测试数据与埋点真值

`agromet/synth.py` 生成三套刻意异构的源：

| 源 | 格式 | 投影/网格 | 不规范点 |
|----|------|-----------|----------|
| A | GRIB2 | Lambert 共形圆锥 20 km，2D 坐标 | cfgrib 短名缺失，按 discipline/category 兜底 |
| B | NetCDF4 | 经纬网格 0.25° | 中文变量名、开尔文、`kg m-2 s-1` 通量 |
| C | NetCDF4 | 旋转极 2D 坐标 0.2° | 无 units 属性、自定义变量名 |

埋入两个已知事件（真值写入 `data/raw/truth_manifest.json`）：

- **旱情 D1**：C03 区县，3/01–3/25 连续 25 个无雨日（日降水 < 0.1 mm）。
- **霜冻 F1**：C01 区县，1/10–1/12 连续 3 天最低温 < -2 ℃（夜间峰值约 -5 ℃）。

验收测试断言判定出的事件时间窗口、持续天数、涉及区县与埋点一致，且邻县
不出现同强度误报。

## 测试

```powershell
py -I -m pytest tests/ -q
```

64 个用例覆盖：单位/变量容错、三源读取、四种插值与权重可复算、面积加权
分区、逐日聚合规则、指标 AST 安全求值、分块管线与内存上限、埋点事件时空
核对、格点级溯源线性还原。

## 关键设计

- **统一稀疏权重矩阵**：四种插值方法都产出行归一化的
  `W (target × source)` 稀疏矩阵并缓存（`regrid.py`）。同一套权重既被
  逐月分块复用，也直接支撑溯源——报告里的源格点权重能严格还原目标值。
- **守恒重采样**：conservative 方法在等积投影（AEA）下计算源/目标格网
  多边形相交面积占比，正确处理跨格点与纬度方向的面积变化。
- **AST 白名单求值**：指标条件表达式只允许比较/布尔/算术节点，拒绝函数
  调用与任意属性访问，不使用 `eval`。
- **内存守卫**：`MemoryGuard` 在 load/regrid/daily 各阶段采样 RSS，超过
  `memory_limit`（默认 4 GiB）立即报错并定位阶段；数据按月切片，权重只
  构建一次。

## 目录结构

```
agromet/
├── agromet/
│   ├── config.py       # 路径/目标网格/内存配置
│   ├── units.py        # 变量名与单位容错识别、换算
│   ├── io.py           # NetCDF/GRIB 读取与标准化
│   ├── regrid.py       # 四种插值 + 稀疏权重 + 溯源
│   ├── zones.py        # 区县/地块面积加权统计
│   ├── resample.py     # 小时→逐日聚合
│   ├── indicators.py   # 可配置灾害指标（AST 安全求值）
│   ├── lineage.py      # 格点级溯源报告
│   ├── pipeline.py     # 分块编排 + 内存守卫
│   ├── synth.py        # 合成异构源与埋点
│   └── web/            # FastAPI 后端 + 静态前端
├── config/             # config.yaml, indicators.yaml
├── scripts/            # 命令行入口
├── tests/              # pytest 测试套件
└── requirements.txt
```
