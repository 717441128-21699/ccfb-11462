"""需求 2: 多网格统一插值, 四种方法 + 溯源权重可复算。"""
import numpy as np
import pytest

from agromet.io import open_dataset
from agromet.regrid import (
    describe_source_grid, build_weights, apply_weights,
    source_contributors, target_cell_index,
)

METHODS = ["nearest", "linear", "cubic", "conservative"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("src_key", ["A", "B", "C"])
def test_regrid_shapes_and_validity(cfg, method, src_key):
    fname = {"A": "source_A_lambert.grib2", "B": "source_B_irregular.nc",
             "C": "source_C_rotated.nc"}[src_key]
    nd = open_dataset(cfg.raw_dir / fname, chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method=method)
    tgt_n = cfg.target_grid.lats.size * cfg.target_grid.lons.size
    assert w.W.shape == (tgt_n, src.n_cells)
    # 行归一化 (有效行)
    sums = np.asarray(w.W.sum(axis=1)).ravel()
    valid_sums = sums[w.target_valid]
    assert valid_sums.size > 0
    np.testing.assert_allclose(valid_sums, 1.0, atol=1e-9)
    # 应用到单时次场
    sample = nd.ds["temp"].isel(time=0).load().values.reshape(-1)
    out = apply_weights(sample, w)
    assert out.shape == (tgt_n,)
    assert np.isfinite(out[w.target_valid]).all()
    nd.ds.close()


def test_nearest_picks_actual_source_value(cfg):
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="nearest")
    sample = nd.ds["temp"].isel(time=0).load().values.reshape(-1)
    out = apply_weights(sample, w)
    # 最近邻: 每个有效输出必须严格等于某个源值
    for t in np.where(w.target_valid)[0][:20]:
        col = w.W.getrow(t).indices[0]
        assert out[t] == pytest.approx(sample[col])
    nd.ds.close()


def test_linear_is_weighted_average(cfg):
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="linear")
    sample = nd.ds["precip"].isel(time=5).load().values.reshape(-1)
    out = apply_weights(sample, w)
    t = int(np.argmax(w.target_valid))
    row = w.W.getrow(t)
    manual = float(sample[row.indices] @ row.data)
    assert out[t] == pytest.approx(manual, rel=1e-10)
    nd.ds.close()


def test_conservation_conservative(cfg):
    """守恒重采样的核心性质: 均匀场重采样后仍是同一常数。

    面积加权权重行归一, 若源场处处为 C, 则每个有效目标格点的
    面积加权平均也必须严格等于 C (与网格分辨率/投影/边缘裁切无关)。
    """
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="conservative")
    for const in (1.0, 7.5, 42.0):
        sample = np.full(src.n_cells, const)
        out = apply_weights(sample, w)
        np.testing.assert_allclose(
            out[w.target_valid], const, rtol=1e-10,
            err_msg=f"守恒重采样未保持均匀场常数 {const}")
    nd.ds.close()


def test_conservative_preserves_spatial_pattern(cfg):
    """非均匀场: 守恒结果应介于源场最小/最大值之间 (权重凸组合)。"""
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="conservative")
    sample = nd.ds["precip"].isel(time=2).load().values.reshape(-1)
    out = apply_weights(sample, w)
    valid = w.target_valid
    assert np.nanmin(out[valid]) >= np.nanmin(sample) - 1e-9
    assert np.nanmax(out[valid]) <= np.nanmax(sample) + 1e-9
    nd.ds.close()


def test_conservative_weights_are_area_fractions(cfg):
    """守恒权重每行是目标格点内各源格点的面积占比, 行和为 1。"""
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="conservative")
    sums = np.asarray(w.W.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums[w.target_valid], 1.0, atol=1e-9)
    # 权重非负
    assert w.W.data.min() >= 0
    nd.ds.close()


def test_lineage_weights_roundtrip(cfg):
    """需求 6: 溯源返回的源格点/权重能严格还原目标值。"""
    nd = open_dataset(cfg.raw_dir / "source_A_lambert.grib2", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="linear")
    sample = nd.ds["temp"].isel(time=3).load().values.reshape(-1)
    t_idx = target_cell_index(w, 30.8, 103.2)
    contribs = source_contributors(w, t_idx)
    manual = sum(c["weight"] * sample[c["source_flat_index"]] for c in contribs)
    out = apply_weights(sample, w)[t_idx]
    assert manual == pytest.approx(out, rel=1e-10)
    assert all(0 < c["weight"] <= 1 for c in contribs)
    nd.ds.close()


def test_weights_cache_roundtrip(tmp_path, cfg):
    nd = open_dataset(cfg.raw_dir / "source_B_irregular.nc", chunks={})
    src = describe_source_grid(nd)
    w = build_weights(src, cfg.target_grid, method="linear")
    w.save(tmp_path, "w1")
    w2 = build_weights(src, cfg.target_grid, method="linear")
    w2_loaded = type(w).load(tmp_path, "w1")
    np.testing.assert_allclose(w.W.toarray(), w2_loaded.W.toarray())
    nd.ds.close()
