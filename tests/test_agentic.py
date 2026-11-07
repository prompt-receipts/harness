import json
from pathlib import Path

import pytest

from receipts_harness import agentic, core

CFG = {"name": "t", "model": "haiku", "runs": 2, "seed": 1, "system_prompt": "s",
       "conditions": {"A": {}, "B": {}}, "tasks": [{"id": "x", "prompt": "do x"}]}


@pytest.fixture
def bases(tmp_path):
    fx = tmp_path / "fx"
    (fx / "app").mkdir(parents=True)
    (fx / "app" / "a.py").write_text("A = 1\n")
    a = agentic.make_base(fx, tmp_path / "baseA")
    b = agentic.make_base(fx, tmp_path / "baseB", {"CLAUDE.md": "rules\n"})
    return {"A": a, "B": b}


def fake_agent(calls):
    def f(prompt, workdir, *, model):
        calls.append((prompt, Path(workdir)))
        (Path(workdir) / "app" / "new.py").write_text("X = 2\n")
        (Path(workdir) / "app" / "__pycache__").mkdir(exist_ok=True)
        (Path(workdir) / "app" / "__pycache__" / "z.pyc").write_text("junk")
        return {"text": "ok", "cost_usd": 0.01, "num_turns": 3, "input_tokens": 1,
                "cache_creation_tokens": 2, "cache_read_tokens": 3, "output_tokens": 4,
                "out_of_tree": [], "events": [{"type": "result"}]}
    return f


def test_condition_files_differ_only_by_claude_md(bases):
    assert not (bases["A"] / "CLAUDE.md").exists()
    assert (bases["B"] / "CLAUDE.md").read_text() == "rules\n"
    assert (bases["A"] / "app" / "a.py").read_bytes() == (bases["B"] / "app" / "a.py").read_bytes()


def test_capture_patch_resume_and_fresh_copies(bases, tmp_path):
    out, calls = tmp_path / "out", []
    n = agentic.run_agentic_capture(CFG, out, bases, agent=fake_agent(calls), version="v", log=lambda *_: None)
    assert n == 4 and len(calls) == 4
    assert len({str(w) for _, w in calls}) == 4          # a fresh workdir per run
    assert all(not w.exists() for _, w in calls)          # and it is removed afterwards
    assert all(p == "do x" for p, _ in calls)
    d = out / "raw" / "B" / "x-0"
    patch = (d / "patch.diff").read_text()
    assert "+++ b/app/new.py" in patch and "CLAUDE.md" not in patch.split("@@")[0].replace("--exclude", "") and "z.pyc" not in patch
    assert json.loads((d / "meta.json").read_text())["cost_usd"] == 0.01
    assert agentic.run_agentic_capture(CFG, out, bases, agent=fake_agent(calls), log=lambda *_: None) == 0
    (d / "meta.json").unlink()
    assert agentic.run_agentic_capture(CFG, out, bases, agent=fake_agent(calls), log=lambda *_: None) == 1


def test_rescore_from_patch(bases, tmp_path):
    out = tmp_path / "out"
    agentic.run_agentic_capture(CFG, out, bases, agent=fake_agent([]), log=lambda *_: None)
    seen = []

    def score(tree, task):
        seen.append((tree / "app" / "new.py").read_text())
        return {"hidden_pass": True}
    rows = agentic.score_agentic(CFG, out, bases, score)
    assert len(rows) == 4 and set(seen) == {"X = 2\n"} and all(r["hidden_pass"] for r in rows)


def test_out_of_tree_flags(tmp_path):
    ev = [{"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/home/x/.claude/CLAUDE.md"}},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "app/a.py"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "cat ~/.claude/CLAUDE.md"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "python -m pytest -q"}}]}}]
    flags = agentic._flag_out_of_tree(ev, tmp_path)
    assert len(flags) == 2 and not any("pytest" in f for f in flags)


def test_clean_env_has_no_claude_session_vars(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "leak")
    env = agentic.clean_env()
    assert "CLAUDE_CODE_SESSION_ID" not in env and env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "--setting-sources" in agentic.AGENT_FLAGS and "project" in agentic.AGENT_FLAGS
