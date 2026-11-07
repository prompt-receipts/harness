# Test receipt

- Tree tested: `6c73d8a9f08e134cc7852afb7ee107abaae554c4`
- Command: `.venv/bin/python -m pytest -q` (Python 3.12.3)
- Result: 16 passed, 0 failed. No live model calls.
- Covers: config, A/B transform, scorer, resume, stats, claude_call parsing, agentic runner (fresh workdir per run, patch capture, resume, rescoring from patch, out-of-tree flags, clean env).
  extraction, scorer pass/fail/timeout/syntax error, resume and partial resume, results
  summary, Fisher exact (known values), Wilson, cluster bootstrap.
