"""农险气象格点数据处理链路 (agromet).

模块:
  config    - 路径/网格/内存配置
  units     - 不规范单位与变量名识别、单位换算
  io        - NetCDF / GRIB 读取与标准化
  regrid    - 多网格统一插值 (nearest/linear/cubic/conservative)
  zones     - 区县/地块面积加权分区统计
  resample  - 逐小时 -> 逐日 (降水累加/温度极值)
  indicators- 可配置灾害指标
  lineage   - 格点级溯源
  pipeline  - 分块编排, 4G 内存约束
  synth     - 合成测试场 (埋旱情/霜冻)
  web       - FastAPI 网页
"""

__version__ = "1.0.0"
