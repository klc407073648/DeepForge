#!/usr/bin/env python3
"""Run local CI checks for a REQ workflow and update state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".workflows"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output
    except FileNotFoundError:
        return 127, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "Command timed out after 300s"


def load_state(req_id: str) -> dict:
    path = WORKFLOWS / req_id / "state.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(req_id: str, state: dict) -> None:
    path = WORKFLOWS / req_id / "state.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CI for REQ workflow")
    parser.add_argument("req_id")
    parser.add_argument("--skip-lint", action="store_true")
    args = parser.parse_args()

    req_id = args.req_id
    wf_dir = WORKFLOWS / req_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    log_path = wf_dir / "ci-last.log"

    state = load_state(req_id)
    from_stage = state.get("stage", "TEST_GEN")
    state["stage"] = "CI_RUNNING"
    state.setdefault("history", []).append(
        {
            "at": utc_now(),
            "action": "ci_start",
            "from_stage": from_stage,
            "to_stage": "CI_RUNNING",
            "by": "run_ci",
        }
    )
    save_state(req_id, state)

    results: list[tuple[str, int, str]] = []
    combined: list[str] = []

    # Lint (ruff if available, else skip gracefully)
    if not args.skip_lint:
        code, out = run_cmd(["python", "-m", "ruff", "check", "src", "tests"], ROOT)
        if code == 127 or "No module named" in out:
            combined.append("[lint] skipped (ruff not installed)\n")
        else:
            results.append(("lint", code, out))
            combined.append(f"=== lint (ruff) exit={code} ===\n{out}\n")

    # Tests (pytest if available)
    code, out = run_cmd(["python", "-m", "pytest", "tests", "-v", "--tb=short"], ROOT)
    if code == 127 or ("No module named" in out and "pytest" in out):
        # Fallback: unittest discovery
        code, out = run_cmd(["python", "-m", "unittest", "discover", "-s", "tests", "-v"], ROOT)
    results.append(("test", code, out))
    combined.append(f"=== test exit={code} ===\n{out}\n")

    log_text = "".join(combined)
    log_path.write_text(log_text, encoding="utf-8")

    all_pass = all(c == 0 for _, c, _ in results if c not in (127,))
    # Treat skipped lint as pass
    effective = all(c == 0 or c == 127 for _, c, _ in results)

    state = load_state(req_id)
    if effective:
        state["stage"] = "CI_RUNNING"
        state["failure_count"] = 0
        action = "ci_pass"
        to_stage = "CI_RUNNING"
        exit_code = 0
    else:
        state["stage"] = "FAILED"
        state["failure_count"] = int(state.get("failure_count", 0)) + 1
        action = "ci_fail"
        to_stage = "FAILED"
        exit_code = 1

    state.setdefault("history", []).append(
        {
            "at": utc_now(),
            "action": action,
            "from_stage": "CI_RUNNING",
            "to_stage": to_stage,
            "by": "run_ci",
        }
    )
    state["last_ci"] = {
        "at": utc_now(),
        "passed": effective,
        "results": [{"name": n, "exit_code": c} for n, c, _ in results],
    }
    save_state(req_id, state)

    print(log_text)
    if not effective:
        print(f"\nCI FAILED for {req_id}. Log: {log_path}")
        print(f"Classify: python scripts/workflow/workflow.py classify-failure {req_id}")
        sys.exit(exit_code)

    print(f"\nCI PASSED for {req_id}")
    print(f"Delivery gate: python scripts/workflow/workflow.py approve {req_id} --gate code --by ci")


if __name__ == "__main__":
    main()
