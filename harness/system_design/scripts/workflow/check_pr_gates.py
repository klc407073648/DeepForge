#!/usr/bin/env python3
"""PR gate: remind reviewers to verify workflow artifacts for REQ branches."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    branch = Path(ROOT / ".git" / "HEAD").read_text(encoding="utf-8") if (ROOT / ".git" / "HEAD").exists() else ""
    req_ids = re.findall(r"REQ-\d+", branch, re.I)

    if not req_ids:
        print("No REQ id in branch name; workflow gate skipped.")
        return

    missing = []
    for req_id in set(req_ids):
        req_id = req_id.upper()
        state = ROOT / ".workflows" / req_id / "state.json"
        matrix = ROOT / "reviews" / f"{req_id}-coverage-matrix.md"
        if not state.exists():
            missing.append(f"{req_id}: missing state.json")
        if not matrix.exists():
            missing.append(f"{req_id}: missing coverage-matrix.md")

    if missing:
        print("Workflow PR gate warnings:")
        for m in missing:
            print(f"  - {m}")
        # Warning only, do not fail CI in pilot phase
        return

    print("Workflow PR gate: all REQ artifacts present.")


if __name__ == "__main__":
    main()
