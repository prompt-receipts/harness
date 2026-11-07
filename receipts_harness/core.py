"""Config, prompt transforms, injectable LLM call, resumable runner, summary."""
import json
import random
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import yaml

from . import stats
from .scorer import score_python

CLAUDE_FLAGS = [
    "-p", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
    "--exclude-dynamic-system-prompt-sections", "--no-session-persistence",
    "--output-format", "json",
]


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    for key in ("name", "model", "runs", "system_prompt", "conditions", "tasks"):
        if key not in cfg:
            raise ValueError(f"config missing '{key}'")
    if set(cfg["conditions"]) != {"A", "B"}:
        raise ValueError("conditions must be exactly A and B")
    cfg["_dir"] = path.parent
    if isinstance(cfg["tasks"], str):  # path to a task list, relative to the config
        cfg["tasks"] = yaml.safe_load((path.parent / cfg["tasks"]).read_text())
    cfg.setdefault("seed", 0)
    cfg.setdefault("timeout", 10)
    return cfg


def lower_first(s):
    return s[:1].lower() + s[1:]


def render(cfg, cond, task):
    """Apply a condition's template. Placeholders: {prompt}, {prompt_lc}."""
    t = cfg["conditions"][cond]["template"]
    return t.format(prompt=task["prompt"], prompt_lc=lower_first(task["prompt"]))


def job_order(cfg):
    """Every (condition, task, run), shuffled with the config seed so A/B interleave."""
    jobs = [(c, t["id"], r) for t in cfg["tasks"] for c in ("A", "B")
            for r in range(cfg["runs"])]
    random.Random(cfg["seed"]).shuffle(jobs)
    return jobs


def claude_version():
    return subprocess.run(["claude", "--version"], capture_output=True,
                          text=True).stdout.strip()


def claude_call(prompt, *, model, system_prompt, timeout=300):
    """Real call: `claude -p` from an empty temp cwd, prompt on stdin."""
    cmd = ["claude", *CLAUDE_FLAGS, "--model", model, "--system-prompt", system_prompt]
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, input=prompt, cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude failed rc={r.returncode}: {r.stderr[:300]}")
    d = json.loads(r.stdout)
    if d.get("is_error"):
        raise RuntimeError(f"claude error: {str(d.get('result'))[:300]}")
    u = d.get("usage", {})
    return {
        "text": d["result"],
        "resolved_model": ",".join(sorted(d.get("modelUsage", {}))),
        "cost_usd": d.get("total_cost_usd", 0.0),
        "input_tokens": u.get("input_tokens", 0),
        "cache_creation_tokens": u.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
    }


def run_capture(cfg, out_dir, llm=claude_call, task_ids=None, runs=None,
                version=None, log=print):
    """Capture raw outputs; skip any already on disk (resumable). Serial."""
    out_dir = Path(out_dir)
    runs = runs if runs is not None else cfg["runs"]
    tasks = {t["id"]: t for t in cfg["tasks"]}
    done = 0
    for cond, tid, r in job_order(cfg):
        if (task_ids and tid not in task_ids) or r >= runs:
            continue
        txt = out_dir / "raw" / cond / f"{tid}-{r}.txt"
        if txt.exists():
            continue
        prompt = render(cfg, cond, tasks[tid])
        res = llm(prompt, model=cfg["model"], system_prompt=cfg["system_prompt"])
        txt.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in res.items() if k != "text"}
        meta.update(model=cfg["model"], cli_version=version, date=date.today().isoformat(),
                    prompt=prompt)
        (txt.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
        txt.write_text(res["text"])  # written last: its presence marks completion
        done += 1
        log(f"{cond} {tid}-{r} in={meta.get('input_tokens')} "
            f"cache_w={meta.get('cache_creation_tokens')} cache_r={meta.get('cache_read_tokens')} "
            f"out={meta.get('output_tokens')} cost={meta.get('cost_usd')}")
    return done


def score_all(cfg, out_dir):
    out_dir = Path(out_dir)
    rows = []
    for t in cfg["tasks"]:
        for c in ("A", "B"):
            for r in range(cfg["runs"]):
                txt = out_dir / "raw" / c / f"{t['id']}-{r}.txt"
                if not txt.exists():
                    continue
                out = txt.read_text()
                meta = json.loads(txt.with_suffix(".json").read_text())
                s = score_python(out, t["tests"], cfg["timeout"])
                rows.append({"condition": c, "task": t["id"], "run": r, "pass": s["pass"],
                             "reason": s["reason"], "output_chars": len(out),
                             "cost_usd": meta.get("cost_usd", 0.0)})
    return rows


def summarize(cfg, rows, version=None):
    per_task = {}
    for x in rows:
        per_task.setdefault(x["task"], {"A": [], "B": []})[x["condition"]].append(int(x["pass"]))
    full = {t: v for t, v in per_task.items() if v["A"] and v["B"]}
    s = {"name": cfg["name"], "model": cfg["model"], "cli_version": version,
         "runs_per_task": cfg["runs"], "tasks": len(cfg["tasks"]), "conditions": {}}
    for c in ("A", "B"):
        rs = [x for x in rows if x["condition"] == c]
        k = sum(x["pass"] for x in rs)
        lo, hi = stats.wilson(k, len(rs))
        s["conditions"][c] = {
            "n": len(rs), "passes": k, "pass_rate": k / len(rs) if rs else None,
            "wilson95": [lo, hi],
            "mean_output_chars": sum(x["output_chars"] for x in rs) / len(rs) if rs else None,
            "cost_usd": sum(x["cost_usd"] for x in rs)}
    a, b = s["conditions"]["A"], s["conditions"]["B"]
    if a["n"] and b["n"]:
        s["fisher_p"] = stats.fisher_exact(a["passes"], a["n"], b["passes"], b["n"])
        d, lo, hi = stats.cluster_bootstrap_diff(full, seed=cfg["seed"])
        s["diff_B_minus_A"] = {"point": d, "cluster_bootstrap95": [lo, hi]}
        s["per_task"] = {t: {c: sum(v[c]) for c in "AB"} for t, v in sorted(per_task.items())}
        s["note"] = ("Pooled Fisher p treats runs as independent; they are not (same tasks). "
                     "The cluster bootstrap resamples tasks and is the more honest interval.")
    s["limits"] = cfg.get("limits", [])
    return s


def write_results(cfg, out_dir, version=None):
    rows = score_all(cfg, out_dir)
    summ = summarize(cfg, rows, version)
    Path(out_dir, "results.json").write_text(
        json.dumps({"summary": summ, "rows": rows}, indent=1))
    return summ
