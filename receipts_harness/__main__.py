"""python -m receipts_harness {run,score} config.yaml [--tasks a,b] [--runs N]"""
import argparse
import json
from pathlib import Path

from .core import claude_version, load_config, run_capture, write_results

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["run", "score"])
ap.add_argument("config")
ap.add_argument("--out", default=None, help="output dir (default: config dir)")
ap.add_argument("--tasks", default=None)
ap.add_argument("--runs", type=int, default=None)
a = ap.parse_args()
cfg = load_config(a.config)
out = Path(a.out) if a.out else cfg["_dir"]
ver = claude_version()
if a.cmd == "run":
    n = run_capture(cfg, out, task_ids=a.tasks.split(",") if a.tasks else None,
                    runs=a.runs, version=ver)
    print(f"captured {n} new outputs")
print(json.dumps(write_results(cfg, out, ver), indent=1))
