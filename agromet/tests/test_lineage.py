"""需求 6: 地块/点位判定的格点级溯源与线性还原。"""
import pandas as pd

from agromet.io import open_dataset
from agromet.regrid import describe_source_grid, build_weights
from agromet.zones import Zone, point_parcel
from agromet.lineage import trace_parcel, save_report, load_report


def test_trace_report_structure_and_reconstruction(cfg):
    dates = pd.date_range("2023-01-10", "2023-01-12", freq="1D")
    weights, datasets = {}, {}
    for fname in ["source_A_lambert.grib2", "source_B_irregular.nc"]:
        stem = fname.rsplit(".", 1)[0]
        nd = open_dataset(cfg.raw_dir / fname, chunks={})
        w = build_weights(describe_source_grid(nd), cfg.target_grid, "linear")
        weights[stem] = w
        datasets[stem] = nd.ds.sel(
            time=slice(dates[0], dates[-1] + pd.Timedelta(hours=23))).load()
        nd.ds.close()
    parcel = point_parcel(103.12, 30.82, "理赔测试点")
    report = trace_parcel(parcel, dates, weights, datasets, "temp")

    assert report["parcel"]["id"] == "理赔测试点"
    assert len(report["target_cells"]) == 1  # 点 -> 单格点
    for sid, s in report["sources"].items():
        assert s["method"] == "linear"
        assert len(s["contributors"]) >= 1
        # 每条贡献都有坐标、权重、小时级原始值
        c0 = s["contributors"][0]["source"]
        assert "lat" in c0 and "lon" in c0 and 0 < c0["weight"] <= 1
        assert len(c0["raw_values"]) == 72  # 3 天 × 24 小时

        # 线性还原: 用溯源权重 × 原始值复算第一个时次的目标值
        from agromet.regrid import apply_weights
        w = weights[sid]
        ds = datasets[sid]
        t_idx = report["target_cells"][0]["cell_index"]
        flat = ds["temp"].isel(time=0).values.reshape(-1)
        expect = apply_weights(flat, w)[t_idx]
        manual = sum(c["source"]["weight"]
                     * c["source"]["raw_values"][0]["value"]
                     for c in s["contributors"]
                     if c["target_cell"]["cell_index"] == t_idx)
        assert abs(manual - expect) < 1e-9


def test_report_save_load_roundtrip(cfg):
    dates = pd.date_range("2023-01-10", periods=2, freq="1D")
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    w = build_weights(describe_source_grid(nd), cfg.target_grid, "nearest")
    ds = nd.ds.sel(time=slice(dates[0], dates[-1] + pd.Timedelta(hours=23))).load()
    report = trace_parcel(
        point_parcel(103.3, 30.7), dates,
        {"source_B_irregular": w}, {"source_B_irregular": ds}, "precip")
    path = save_report(report, cfg.claims_dir, "unit_test_report")
    again = load_report(path)
    assert again["variable"] == "precip"
    assert again["sources"].keys() == report["sources"].keys()
    nd.ds.close()


def _stem(fname: str) -> str:
    return fname.rsplit(".", 1)[0]
