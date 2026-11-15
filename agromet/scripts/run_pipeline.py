"""一键跑通整条链路的命令行入口。

用法:
  py -I scripts/run_pipeline.py synth          # 生成合成数据
  py -I scripts/run_pipeline.py run --method linear
  py -I scripts/run_pipeline.py all --method linear   # 合成 + 全流程
  py -I scripts/run_pipeline.py web            # 启动网页
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agromet.config import Config
from agromet.synth import generate_all
from agromet.pipeline import AgrometPipeline
from agromet.io import GRIB_SUFFIXES, NETCDF_SUFFIXES


def main():
    ap = argparse.ArgumentParser(description="农险气象处理链路")
    ap.add_argument("action", choices=["synth", "run", "all", "web"])
    ap.add_argument("--method", default="linear",
                    choices=["nearest", "linear", "cubic", "conservative"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cfg = Config.load()

    if args.action in ("synth", "all"):
        print("[1/2] 生成合成格点场 (一年逐小时, 三套异构源)...")
        manifest = generate_all(cfg)
        print(f"      数据源 {len(manifest['sources'])} 个, "
              f"埋入事件 {len(manifest['events'])} 个")
        for ev in manifest["events"]:
            print(f"      - {ev['hazard']} {ev['event_id']}: {ev['start']} ~ "
                  f"{ev['end']} @ {','.join(ev['counties'])}")

    if args.action in ("run", "all"):
        print("[2/2] 逐月分块处理...")
        pipe = AgrometPipeline(cfg)
        files = [p for p in sorted(cfg.raw_dir.iterdir())
                 if p.suffix.lower() in GRIB_SUFFIXES | NETCDF_SUFFIXES]
        for f in files:
            print(f"  >> {f.name} (method={args.method})")
            task = pipe.run_source(f, method=args.method)
            peak = max((m["rss_mb"] for m in task.memory_trace), default=0)
            print(f"     status={task.status} chunks={task.chunks} "
                  f"peak_rss={peak}MB outputs={len(task.outputs)}")
            if task.status == "failed":
                print(f"     ERROR: {task.error}")
                sys.exit(1)

    if args.action == "web":
        import uvicorn
        uvicorn.run("agromet.web.app:app", host=args.host, port=args.port,
                    reload=False)


if __name__ == "__main__":
    main()
