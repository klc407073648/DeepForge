#!/usr/bin/env python3
"""Cursor hook: inject workflow context and gate code edits without approval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".workflows"

PROTECTED_PREFIXES = ("src/", "tests/")
IMPLEMENT_BLOCKED_STAGES = {
    "DRAFT_PLAN",
    "PLAN_REVIEW",
    "DIAGRAM_DRAFT",
}


def active_workflows() -> list[dict]:
    if not WORKFLOWS.exists():
        return []
    active = []
    for d in WORKFLOWS.iterdir():
        state_file = d / "state.json"
        if not state_file.exists():
            continue
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("stage") not in ("PASSED",):
                active.append(state)
        except (json.JSONDecodeError, OSError):
            continue
    return active


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    active = active_workflows()
    if not active:
        print("{}")
        return

    lines = ["[Workflow Gate] Active closed-loop workflows:"]
    blocked = False
    for state in active:
        req_id = state.get("req_id", "?")
        stage = state.get("stage", "?")
        approvals = list(state.get("approvals", {}).keys())
        lines.append(f"- {req_id}: stage={stage}, approvals={approvals}")

        if stage in IMPLEMENT_BLOCKED_STAGES:
            blocked = True

    context = "\n".join(lines)
    if blocked:
        context += (
            "\n\nCode generation is gated until plan and diagram are APPROVED. "
            "Use: python scripts/workflow/workflow.py validate REQ-xxx --action implement"
        )

    print(json.dumps({"additional_context": context}))


if __name__ == "__main__":
    main()
