import json
import textwrap

import pytest

from receipts_harness import core, stats
from receipts_harness.scorer import extract_code, score_python

CFG = textwrap.dedent("""
    name: t
    model: haiku
    runs: 3
    seed: 1
    system_prompt: sys
    conditions:
      A: {template: "{prompt}"}
      B: {template: "Please {prompt_lc} Thank you."}
    tasks:
      - id: add
        prompt: Write add(a, b).
        tests: "assert add(1, 2) == 3"
      - id: neg
        prompt: Write neg(a).
        tests: "assert neg(1) == -1"
""")


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(CFG)
    return core.load_config(p)


def test_config_load(cfg):
    assert cfg["model"] == "haiku" and len(cfg["tasks"]) == 2 and cfg["timeout"] == 10


def test_config_rejects_missing_and_bad_conditions(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\n")
    with pytest.raises(ValueError):
        core.load_config(p)
    p.write_text(CFG.replace("  B:", "  C:"))
    with pytest.raises(ValueError):
        core.load_config(p)


def test_ab_transform_only_differs_in_wording(cfg):
    t = cfg["tasks"][0]
    assert core.render(cfg, "A", t) == "Write add(a, b)."
    assert core.render(cfg, "B", t) == "Please write add(a, b). Thank you."


def test_job_order_complete_deterministic_interleaved(cfg):
    jobs = core.job_order(cfg)
    assert len(jobs) == 12 and len(set(jobs)) == 12
    assert jobs == core.job_order(cfg)
    assert jobs != sorted(jobs)


def test_extract_code():
    assert extract_code("hi\n```python\nx=1\n```\nbye").strip() == "x=1"
    assert extract_code("```\na\n```\n```python\nb\n```").strip() == "b"
    assert extract_code("x = 2") == "x = 2"


def test_scorer_pass_fail_timeout():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert score_python(good, "assert add(1, 2) == 3")["pass"]
    bad = "```python\ndef add(a, b):\n    return 0\n```"
    r = score_python(bad, "assert add(1, 2) == 3")
    assert not r["pass"] and "AssertionError" in r["reason"]
    hang = "```python\ndef add(a, b):\n    while True: pass\n```"
    assert score_python(hang, "add(1, 2)", timeout=1) == {"pass": False, "reason": "timeout"}
    assert not score_python("def add(:", "add(1,2)")["pass"]


def fake_llm(calls):
    def f(prompt, *, model, system_prompt):
        calls.append(prompt)
        body = "-a" if "neg" in prompt else "a + b"
        return {"text": f"```python\ndef add(a, b):\n    return {body}\ndef neg(a):\n    return {body}\n```",
                "cost_usd": 0.01, "input_tokens": 3, "output_tokens": 5,
                "cache_creation_tokens": 0, "cache_read_tokens": 0}
    return f


def test_resume_skips_captured(cfg, tmp_path):
    out = tmp_path / "out"
    calls = []
    assert core.run_capture(cfg, out, llm=fake_llm(calls), version="v", log=lambda *_: None) == 12
    assert len(calls) == 12
    meta = json.loads((out / "raw/A/add-0.json").read_text())
    assert meta["model"] == "haiku" and meta["cli_version"] == "v" and meta["cost_usd"] == 0.01
    assert core.run_capture(cfg, out, llm=fake_llm(calls), log=lambda *_: None) == 0
    assert len(calls) == 12
    (out / "raw/B/neg-2.txt").unlink()
    assert core.run_capture(cfg, out, llm=fake_llm(calls), log=lambda *_: None) == 1


def test_subset_and_end_to_end_results(cfg, tmp_path):
    out = tmp_path / "out"
    calls = []
    core.run_capture(cfg, out, llm=fake_llm(calls), task_ids={"add"}, runs=1, log=lambda *_: None)
    assert len(calls) == 2
    core.run_capture(cfg, out, llm=fake_llm(calls), log=lambda *_: None)
    s = core.write_results(cfg, out, "v")
    assert s["conditions"]["A"]["n"] == 6 and s["conditions"]["A"]["pass_rate"] == 1.0
    assert s["conditions"]["B"]["cost_usd"] == pytest.approx(0.06)
    assert s["diff_B_minus_A"]["point"] == 0 and s["fisher_p"] == 1.0
    assert (out / "results.json").exists()


def test_stats():
    assert stats.fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)
    assert stats.fisher_exact(10, 10, 0, 10) == pytest.approx(2 / 184756)
    # known value: 8/10 vs 3/10 -> p ~ 0.0698
    assert stats.fisher_exact(8, 10, 3, 10) == pytest.approx(0.0698, abs=1e-3)
    lo, hi = stats.wilson(5, 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert stats.wilson(0, 0) == (0.0, 0.0)
    pt = {"t1": {"A": [1, 1], "B": [0, 0]}, "t2": {"A": [1, 1], "B": [1, 1]}}
    d, lo, hi = stats.cluster_bootstrap_diff(pt, iters=500, seed=3)
    assert d == -0.5 and lo <= d <= hi


def test_tasks_file_reference(tmp_path):
    (tmp_path / "tasks.yaml").write_text("- id: a\n  prompt: P\n  tests: pass\n")
    (tmp_path / "c.yaml").write_text(CFG.split("tasks:")[0] + "tasks: tasks.yaml\n")
    assert core.load_config(tmp_path / "c.yaml")["tasks"][0]["id"] == "a"


def test_claude_call_parses_json(monkeypatch):
    out = json.dumps({"result": "hi", "total_cost_usd": 0.5, "is_error": False,
                      "modelUsage": {"claude-x": {}}, "usage": {"input_tokens": 2, "output_tokens": 3}})
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["input"] = cmd, kw["input"]
        class R: returncode, stdout, stderr = 0, out, ""
        return R
    monkeypatch.setattr(core.subprocess, "run", fake_run)
    r = core.claude_call("P", model="haiku", system_prompt="S")
    assert r["text"] == "hi" and r["cost_usd"] == 0.5 and r["resolved_model"] == "claude-x"
    assert seen["input"] == "P" and "--tools" in seen["cmd"] and "--no-session-persistence" in seen["cmd"]
