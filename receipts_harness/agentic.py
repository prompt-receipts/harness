"""Agentic runs: the model edits a fresh copy of a fixture repo; results are saved as a patch.

Isolation: the child gets a minimal environment, loads only project-level settings
(`--setting-sources project`, so the user's global CLAUDE.md, hooks and settings are not read),
no MCP servers, and no auto memory. Scoring rebuilds the tree from base + saved patch, so any
result can be re-scored without a model.
"""
import json
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from . import core

TOOLS = "Read,Edit,Write,Bash"
AGENT_FLAGS = ["-p", "--tools", TOOLS, "--allowedTools", TOOLS, "--setting-sources", "project",
               "--strict-mcp-config", "--exclude-dynamic-system-prompt-sections",
               "--disable-slash-commands", "--no-session-persistence",
               "--output-format", "stream-json", "--verbose"]
DIFF_EXCLUDES = ["__pycache__", ".pytest_cache", "*.pyc"]


def clean_env():
    env = {"HOME": os.environ["HOME"], "PATH": os.environ.get("PATH", ""),
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "LANG": "C.UTF-8"}
    return env


def make_base(fixture, dest, extra_files=None):
    """dest = copy of fixture plus extra {relative path: text} files (e.g. CLAUDE.md)."""
    dest = Path(dest)
    shutil.copytree(fixture, dest, ignore=shutil.ignore_patterns(*DIFF_EXCLUDES))
    for rel, text in (extra_files or {}).items():
        (dest / rel).write_text(text)
    return dest


def make_patch(base, work):
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(base, Path(d) / "a")
        shutil.copytree(work, Path(d) / "b", ignore=shutil.ignore_patterns(*DIFF_EXCLUDES))
        cmd = ["diff", "-ruN"] + [f"--exclude={x}" for x in DIFF_EXCLUDES] + ["a", "b"]
        return subprocess.run(cmd, cwd=d, capture_output=True, text=True).stdout


def rebuild_tree(base, patch_text, dest):
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns(*DIFF_EXCLUDES))
    if patch_text.strip():
        r = subprocess.run(["patch", "-p1", "-s", "-d", str(dest)], input=patch_text,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"patch failed: {r.stderr[:300]}")
    return dest


def _flag_out_of_tree(events, workdir):
    """Tool calls that reach outside the workdir or toward the user's Claude config."""
    root = os.path.realpath(workdir)
    flags = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for blk in ev.get("message", {}).get("content", []):
            if blk.get("type") != "tool_use":
                continue
            inp = blk.get("input", {})
            p = inp.get("file_path") or inp.get("path")
            if p and not os.path.realpath(os.path.join(root, p)).startswith(root):
                flags.append(f"{blk['name']} {p}")
            cmd = inp.get("command", "")
            if any(x in cmd for x in (".claude", "/home/", "~/", "$HOME")):
                flags.append(f"Bash {cmd[:120]}")
    return flags


def agent_call(prompt, workdir, *, model, max_budget_usd=0.5, timeout=600):
    cmd = ["claude", *AGENT_FLAGS, "--model", model, "--max-budget-usd", str(max_budget_usd)]
    r = subprocess.run(cmd, input=prompt, cwd=workdir, env=clean_env(), capture_output=True,
                       text=True, timeout=timeout)
    events = []
    for line in r.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    final = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if final is None:
        raise RuntimeError(f"no result event rc={r.returncode}: {r.stderr[:300]}")
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    u = final.get("usage", {})
    return {
        "text": final.get("result", ""), "is_error": final.get("is_error", False),
        "cost_usd": final.get("total_cost_usd", 0.0), "num_turns": final.get("num_turns"),
        "input_tokens": u.get("input_tokens", 0),
        "cache_creation_tokens": u.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "resolved_model": ",".join(sorted(final.get("modelUsage", {}))),
        "out_of_tree": _flag_out_of_tree(events, workdir),
        "init": {k: init.get(k) for k in ("tools", "mcp_servers", "plugins", "skills", "agents",
                                          "memory_paths", "permissionMode", "cwd")},
        "events": events,
    }


def run_agentic_capture(cfg, out_dir, bases, agent=agent_call, task_ids=None, runs=None,
                        version=None, log=print):
    """Serial, resumable. bases: {"A": dir, "B": dir}. Saves patch + meta + transcript per run."""
    out_dir = Path(out_dir)
    runs = runs if runs is not None else cfg["runs"]
    tasks = {t["id"]: t for t in cfg["tasks"]}
    done = 0
    for cond, tid, r in core.job_order(cfg):
        if (task_ids and tid not in task_ids) or r >= runs:
            continue
        d = out_dir / "raw" / cond / f"{tid}-{r}"
        if (d / "meta.json").exists():
            continue
        prompt = tasks[tid]["prompt"]
        with tempfile.TemporaryDirectory(prefix="pr2-") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(bases[cond], work)
            res = agent(prompt, work, model=cfg["model"])
            patch = make_patch(bases[cond], work)
        d.mkdir(parents=True, exist_ok=True)
        events = res.pop("events", [])
        (d / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        (d / "patch.diff").write_text(patch)
        meta = {k: v for k, v in res.items()}
        meta.update(model=cfg["model"], cli_version=version, date=date.today().isoformat(),
                    prompt=prompt)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))  # last: marks completion
        done += 1
        log(f"{cond} {tid}-{r} turns={meta.get('num_turns')} cost={meta.get('cost_usd'):.4f} "
            f"out={meta.get('output_tokens')} cache_w={meta.get('cache_creation_tokens')} "
            f"cache_r={meta.get('cache_read_tokens')} oot={len(meta.get('out_of_tree', []))}")
    return done


def score_agentic(cfg, out_dir, bases, score_tree):
    """Rebuild each saved run from base + patch and score it with score_tree(tree, task)."""
    out_dir = Path(out_dir)
    rows = []
    for t in cfg["tasks"]:
        for c in ("A", "B"):
            for r in range(cfg["runs"]):
                d = out_dir / "raw" / c / f"{t['id']}-{r}"
                if not (d / "meta.json").exists():
                    continue
                meta = json.loads((d / "meta.json").read_text())
                with tempfile.TemporaryDirectory() as tmp:
                    tree = rebuild_tree(bases[c], (d / "patch.diff").read_text(), Path(tmp) / "t")
                    s = score_tree(tree, t)
                rows.append({"condition": c, "task": t["id"], "run": r, **s,
                             "cost_usd": meta.get("cost_usd", 0.0),
                             "turns": meta.get("num_turns"),
                             "out_of_tree": meta.get("out_of_tree", [])})
    return rows
