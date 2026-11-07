"""Deterministic scorer: run hidden unit tests against extracted code."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text):
    """Last fenced block, or the whole text if there is no fence."""
    blocks = _BLOCK.findall(text)
    return blocks[-1] if blocks else text


def score_python(output, tests, timeout=10):
    """Return {"pass": bool, "reason": str}. `tests` is Python source of asserts."""
    code = extract_code(output)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "run.py"
        p.write_text(code + "\n\n" + tests + "\n")
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(p)], cwd=d, capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"pass": False, "reason": "timeout"}
    if r.returncode == 0:
        return {"pass": True, "reason": "ok"}
    return {"pass": False, "reason": (r.stderr.strip().splitlines() or ["error"])[-1][:200]}
