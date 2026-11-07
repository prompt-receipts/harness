# receipts harness

Runs the same small tasks under two prompt conditions (A and B), many times, scores each
output with hidden unit tests, and reports pass rates with a significance note and limits.
It is the engine behind the receipts in the `prompt-receipts` account.

## Setup

```
git clone https://github.com/prompt-receipts/harness
cd harness
python3.12 -m venv .venv && .venv/bin/pip install pyyaml pytest
.venv/bin/python -m pytest -q      # 16 tests, no model calls
```

You also need the `claude` CLI installed and logged in to run real experiments.

## Use

A receipt repo (for example `receipt-002-claude-md`) sits next to this directory as a sibling and points at
this code with `PYTHONPATH=../harness`:

- `python -m receipts_harness run config.yaml [--tasks a,b] [--runs N]` captures raw outputs
  (resumable: existing files are skipped) and writes `results.json`.
- `python -m receipts_harness score config.yaml` rescores what is on disk, with no model calls.
- `receipts_harness/agentic.py` runs the model as an agent on a fresh copy of a fixture repo, saves each
  result as a patch, and rescores from the patch.
- Plain calls go through `claude -p` from an empty temp directory with tools, settings and MCP off.
  Agentic calls load only project settings, so a personal global CLAUDE.md is not read.

`RECEIPT.md` names the commit the tests last passed on. MIT licensed.
