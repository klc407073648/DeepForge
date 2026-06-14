#!/usr/bin/env python3
"""Closed-loop workflow orchestrator CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
WORKFLOWS = ROOT / ".workflows"

STAGES = [
    "DRAFT_PLAN",
    "PLAN_REVIEW",
    "DIAGRAM_DRAFT",
    "DIAGRAM_APPROVED",
    "CODE_GEN",
    "TEST_GEN",
    "CI_RUNNING",
    "PASSED",
    "FAILED",
]

STAGE_TRANSITIONS: dict[str, list[str]] = {
    "DRAFT_PLAN": ["PLAN_REVIEW"],
    "PLAN_REVIEW": ["DIAGRAM_DRAFT", "DRAFT_PLAN"],
    "DIAGRAM_DRAFT": ["DIAGRAM_APPROVED", "PLAN_REVIEW"],
    "DIAGRAM_APPROVED": ["CODE_GEN"],
    "CODE_GEN": ["TEST_GEN", "DIAGRAM_DRAFT"],
    "TEST_GEN": ["CI_RUNNING", "CODE_GEN"],
    "CI_RUNNING": ["PASSED", "FAILED"],
    "FAILED": ["CODE_GEN", "TEST_GEN", "PLAN_REVIEW", "DIAGRAM_DRAFT"],
    "PASSED": [],
}

GATE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "plan": {"required_stage": "PLAN_REVIEW", "next_stage": "DIAGRAM_DRAFT"},
    "diagram": {"required_stage": "DIAGRAM_DRAFT", "next_stage": "DIAGRAM_APPROVED"},
    "code": {"required_stage": "CI_RUNNING", "next_stage": "PASSED"},
}

ACTION_GATES: dict[str, dict[str, Any]] = {
    "implement": {
        "allowed_stages": {"DIAGRAM_APPROVED", "CODE_GEN", "TEST_GEN"},
        "required_approvals": {"plan", "diagram"},
    },
    "test": {
        "allowed_stages": {"CODE_GEN", "TEST_GEN", "CI_RUNNING", "FAILED"},
        "required_approvals": {"plan", "diagram"},
    },
    "diagram": {
        "allowed_stages": {"PLAN_REVIEW", "DIAGRAM_DRAFT"},
        "required_approvals": {"plan"},
    },
}

NEXT_SKILL: dict[str, str] = {
    "DRAFT_PLAN": "workflow-requirement → workflow-architect",
    "PLAN_REVIEW": "人工审核方案 (checklists/plan-review.md) → approve --gate plan",
    "DIAGRAM_DRAFT": "workflow-diagram → 人工确认 → approve --gate diagram",
    "DIAGRAM_APPROVED": "workflow-implement",
    "CODE_GEN": "workflow-implement (继续) 或 advance --to TEST_GEN",
    "TEST_GEN": "workflow-tdd",
    "CI_RUNNING": "等待 CI 或 run_ci.py",
    "PASSED": "创建 PR，流程完成",
    "FAILED": "workflow-fix → classify-failure → rollback",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_path(req_id: str) -> Path:
    return WORKFLOWS / req_id / "state.json"


def load_state(req_id: str) -> dict[str, Any]:
    path = state_path(req_id)
    if not path.exists():
        raise SystemExit(f"Workflow not found: {req_id}. Run: workflow.py init {req_id}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(req_id: str, state: dict[str, Any]) -> None:
    path = state_path(req_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_history(
    state: dict[str, Any],
    action: str,
    from_stage: str | None,
    to_stage: str,
    by: str = "system",
    reason: str = "",
) -> None:
    entry: dict[str, Any] = {
        "at": utc_now(),
        "action": action,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "by": by,
    }
    if reason:
        entry["reason"] = reason
    state.setdefault("history", []).append(entry)


def replace_template(text: str, req_id: str, **extra: str) -> str:
    result = text.replace("REQ-XXX", req_id)
    for key, value in extra.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def copy_template(template_name: str, dest: Path, req_id: str, **extra: str) -> None:
    src = TEMPLATES / template_name
    if not src.exists():
        return
    content = replace_template(src.read_text(encoding="utf-8"), req_id, **extra)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> None:
    req_id = args.req_id
    wf_dir = WORKFLOWS / req_id
    if (wf_dir / "state.json").exists() and not args.force:
        raise SystemExit(f"Workflow {req_id} already exists. Use --force to re-init.")

    if wf_dir.exists() and args.force:
        shutil.rmtree(wf_dir)

    wf_dir.mkdir(parents=True, exist_ok=True)

    state = json.loads(
        replace_template((TEMPLATES / "state.json").read_text(encoding="utf-8"), req_id)
    )
    state["req_id"] = req_id
    state["source_url"] = args.url or ""
    state["stage"] = "DRAFT_PLAN"
    state["history"] = [
        {
            "at": utc_now(),
            "action": "init",
            "from_stage": None,
            "to_stage": "DRAFT_PLAN",
            "by": args.by or "system",
        }
    ]
    save_state(req_id, state)

    copy_template("requirement.md", ROOT / "requirements" / f"{req_id}.md", req_id, title=args.title or "")
    copy_template("plan.md", ROOT / "plans" / f"{req_id}-plan-v1.md", req_id, title=args.title or "")
    copy_template("review.md", ROOT / "reviews" / f"{req_id}-plan-review.md", req_id)
    (ROOT / "diagrams" / req_id).mkdir(parents=True, exist_ok=True)
    copy_template("diagram-readme.md", ROOT / "diagrams" / req_id / "README.md", req_id)

    print(f"Initialized workflow {req_id}")
    print(f"  state: {state_path(req_id)}")
    print(f"  next:  {NEXT_SKILL['DRAFT_PLAN']}")


def cmd_status(args: argparse.Namespace) -> None:
    state = load_state(args.req_id)
    if getattr(args, "human", False):
        sys.path.insert(0, str(ROOT / "src"))
        from workflow.status_formatter import format_status_summary

        print(format_status_summary(state, NEXT_SKILL))
    else:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        print(f"\nNext step: {NEXT_SKILL.get(state['stage'], 'unknown')}")


def cmd_next(args: argparse.Namespace) -> None:
    state = load_state(args.req_id)
    stage = state["stage"]
    print(f"REQ: {args.req_id}")
    print(f"Stage: {stage}")
    print(f"Next: {NEXT_SKILL.get(stage, 'unknown')}")


def _transition(state: dict[str, Any], to_stage: str, by: str, reason: str = "") -> None:
    from_stage = state["stage"]
    allowed = STAGE_TRANSITIONS.get(from_stage, [])
    if to_stage not in allowed and from_stage != to_stage:
        raise SystemExit(
            f"Invalid transition: {from_stage} → {to_stage}. Allowed: {allowed}"
        )
    append_history(state, "advance", from_stage, to_stage, by, reason)
    state["stage"] = to_stage


def cmd_advance(args: argparse.Namespace) -> None:
    state = load_state(args.req_id)
    _transition(state, args.to, args.by or "user", args.reason or "")
    save_state(args.req_id, state)
    print(f"Advanced {args.req_id}: → {args.to}")
    print(f"Next: {NEXT_SKILL.get(args.to, 'unknown')}")


def cmd_approve(args: argparse.Namespace) -> None:
    gate = args.gate
    if gate not in GATE_REQUIREMENTS:
        raise SystemExit(f"Unknown gate: {gate}. Use: plan, diagram, code")

    state = load_state(args.req_id)
    cfg = GATE_REQUIREMENTS[gate]
    if state["stage"] != cfg["required_stage"]:
        raise SystemExit(
            f"Cannot approve gate '{gate}' at stage '{state['stage']}'. "
            f"Required: {cfg['required_stage']}"
        )

    state.setdefault("approvals", {})[gate] = {
        "by": args.by or "user",
        "at": utc_now(),
    }
    _transition(state, cfg["next_stage"], args.by or "user", f"approved gate: {gate}")
    save_state(args.req_id, state)

    review_type = {"plan": "plan", "diagram": "diagram", "code": "code"}.get(gate, gate)
    review_path = ROOT / "reviews" / f"{args.req_id}-{review_type}-review.md"
    if review_path.exists():
        text = review_path.read_text(encoding="utf-8")
        text = re.sub(r"status:\s*\w+", "status: APPROVED", text, count=1)
        text = text.replace("- [ ] APPROVED", "- [x] APPROVED")
        text = text.replace("- [x] REJECTED", "- [ ] REJECTED")
        review_path.write_text(text, encoding="utf-8")

    print(f"Approved gate '{gate}' for {args.req_id} → {cfg['next_stage']}")


def cmd_reject(args: argparse.Namespace) -> None:
    gate = args.gate
    state = load_state(args.req_id)

    if gate == "plan":
        if state["stage"] != "PLAN_REVIEW":
            raise SystemExit("Plan rejection only valid at PLAN_REVIEW")
        state["plan_version"] = int(state.get("plan_version", 1)) + 1
        _transition(state, "DRAFT_PLAN", args.by or "user", args.reason or "plan rejected")
        review_path = ROOT / "reviews" / f"{args.req_id}-plan-review.md"
    elif gate == "diagram":
        if state["stage"] != "DIAGRAM_DRAFT":
            raise SystemExit("Diagram rejection only valid at DIAGRAM_DRAFT")
        state["diagram_version"] = int(state.get("diagram_version", 1)) + 1
        _transition(state, "PLAN_REVIEW", args.by or "user", args.reason or "diagram rejected")
        review_path = ROOT / "reviews" / f"{args.req_id}-diagram-review.md"
    else:
        raise SystemExit("Reject supports: plan, diagram")

    save_state(args.req_id, state)

    if review_path.exists():
        text = review_path.read_text(encoding="utf-8")
        text = re.sub(r"status:\s*\w+", "status: REJECTED", text, count=1)
        text = text.replace("- [ ] REJECTED", "- [x] REJECTED")
        text = text.replace("- [x] APPROVED", "- [ ] APPROVED")
        if args.reason:
            text += f"\n\n## 驳回原因\n\n{args.reason}\n"
        review_path.write_text(text, encoding="utf-8")

    print(f"Rejected gate '{gate}' for {args.req_id}. Reason: {args.reason or '(none)'}")


def cmd_validate(args: argparse.Namespace) -> None:
    state = load_state(args.req_id)
    action = args.action
    cfg = ACTION_GATES.get(action)
    if not cfg:
        raise SystemExit(f"Unknown action: {action}. Use: implement, test, diagram")

    stage = state["stage"]
    approvals = set(state.get("approvals", {}).keys())
    required = set(cfg["required_approvals"])

    errors: list[str] = []
    if stage not in cfg["allowed_stages"]:
        errors.append(f"stage '{stage}' not in allowed {cfg['allowed_stages']}")
    missing = required - approvals
    if missing:
        errors.append(f"missing approvals: {missing}")

    if errors:
        print(json.dumps({"valid": False, "req_id": args.req_id, "action": action, "errors": errors}, indent=2))
        sys.exit(1)

    print(json.dumps({"valid": True, "req_id": args.req_id, "action": action, "stage": stage}, indent=2))


def cmd_rollback(args: argparse.Namespace) -> None:
    state = load_state(args.req_id)
    to_stage = args.to
    if to_stage not in STAGES:
        raise SystemExit(f"Invalid stage: {to_stage}")

    state["failure_count"] = int(state.get("failure_count", 0)) + 1
    max_fail = int(state.get("max_failures_before_escalate", 3))
    if state["failure_count"] > max_fail:
        print(f"ESCALATE: failure_count {state['failure_count']} > {max_fail}. Manual intervention required.")
        state["stage"] = "FAILED"
        append_history(state, "escalate", state["stage"], "FAILED", args.by or "system", args.reason or "")
        save_state(args.req_id, state)
        sys.exit(2)

    from_stage = state["stage"]
    append_history(state, "rollback", from_stage, to_stage, args.by or "system", args.reason or "")
    state["stage"] = to_stage
    save_state(args.req_id, state)
    print(f"Rolled back {args.req_id}: {from_stage} → {to_stage}")
    print(f"Next: {NEXT_SKILL.get(to_stage, 'unknown')}")


FAILURE_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("test_assertion", "TEST_GEN", re.compile(r"AssertionError|assert.*failed|Expected.*but got", re.I)),
    ("test_import", "TEST_GEN", re.compile(r"ImportError|ModuleNotFoundError|cannot import", re.I)),
    ("test_syntax", "TEST_GEN", re.compile(r"SyntaxError.*test_", re.I)),
    ("lint", "CODE_GEN", re.compile(r"lint|flake8|ruff|eslint|pylint", re.I)),
    ("type_check", "CODE_GEN", re.compile(r"mypy|type error|TypeError(?!.*test_)", re.I)),
    ("logic", "CODE_GEN", re.compile(r"AttributeError|KeyError|ValueError|IndexError", re.I)),
    ("architecture", "DIAGRAM_DRAFT", re.compile(r"circular import|module not found.*src", re.I)),
    ("plan_gap", "PLAN_REVIEW", re.compile(r"not implemented|TODO|missing feature|scope", re.I)),
]


def classify_log(log_text: str) -> dict[str, str]:
    for name, stage, pattern in FAILURE_PATTERNS:
        if pattern.search(log_text):
            return {
                "classification": name,
                "rollback_stage": stage,
                "root_cause_hint": f"Matched pattern: {name}",
            }
    return {
        "classification": "unknown",
        "rollback_stage": "CODE_GEN",
        "root_cause_hint": "Unable to classify; defaulting to CODE_GEN",
    }


def cmd_classify_failure(args: argparse.Namespace) -> None:
    log_path = Path(args.log) if args.log else WORKFLOWS / args.req_id / "ci-last.log"
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    result = classify_log(log_text)
    result["req_id"] = args.req_id
    result["log_path"] = str(log_path)

    out_path = WORKFLOWS / args.req_id / "last-classification.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    state = load_state(args.req_id)
    state["stage"] = "FAILED"
    state["last_classification"] = result
    append_history(
        state,
        "classify_failure",
        state["stage"],
        "FAILED",
        "system",
        result["classification"],
    )
    save_state(args.req_id, state)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSuggested rollback:")
    print(f"  python scripts/workflow/workflow.py rollback {args.req_id} --to {result['rollback_stage']}")


def cmd_list(args: argparse.Namespace) -> None:
    if not WORKFLOWS.exists():
        print("No workflows.")
        return
    rows = []
    for d in sorted(WORKFLOWS.iterdir()):
        if d.is_dir() and (d / "state.json").exists():
            state = json.loads((d / "state.json").read_text(encoding="utf-8"))
            rows.append((state["req_id"], state["stage"], state.get("plan_version", 1)))
    if not rows:
        print("No workflows.")
        return
    print(f"{'REQ_ID':<12} {'STAGE':<20} {'PLAN_VER'}")
    print("-" * 44)
    for req_id, stage, ver in rows:
        print(f"{req_id:<12} {stage:<20} {ver}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop workflow orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a new REQ workflow")
    p_init.add_argument("req_id")
    p_init.add_argument("--url", default="", help="Requirement document URL")
    p_init.add_argument("--title", default="", help="Requirement title")
    p_init.add_argument("--by", default="user")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="Show workflow state")
    p_status.add_argument("req_id")
    p_status.add_argument("--human", action="store_true", help="Human-readable summary")
    p_status.set_defaults(func=cmd_status)

    p_next = sub.add_parser("next", help="Show next recommended step")
    p_next.add_argument("req_id")
    p_next.set_defaults(func=cmd_next)

    p_adv = sub.add_parser("advance", help="Advance to a stage")
    p_adv.add_argument("req_id")
    p_adv.add_argument("--to", required=True)
    p_adv.add_argument("--by", default="user")
    p_adv.add_argument("--reason", default="")
    p_adv.set_defaults(func=cmd_advance)

    p_apr = sub.add_parser("approve", help="Approve a review gate")
    p_apr.add_argument("req_id")
    p_apr.add_argument("--gate", required=True, choices=["plan", "diagram", "code"])
    p_apr.add_argument("--by", default="user")
    p_apr.set_defaults(func=cmd_approve)

    p_rej = sub.add_parser("reject", help="Reject a review gate")
    p_rej.add_argument("req_id")
    p_rej.add_argument("--gate", required=True, choices=["plan", "diagram"])
    p_rej.add_argument("--by", default="user")
    p_rej.add_argument("--reason", default="")
    p_rej.set_defaults(func=cmd_reject)

    p_val = sub.add_parser("validate", help="Validate if an action is allowed")
    p_val.add_argument("req_id")
    p_val.add_argument("--action", required=True, choices=["implement", "test", "diagram"])
    p_val.set_defaults(func=cmd_validate)

    p_rb = sub.add_parser("rollback", help="Rollback to a previous stage")
    p_rb.add_argument("req_id")
    p_rb.add_argument("--to", required=True)
    p_rb.add_argument("--by", default="system")
    p_rb.add_argument("--reason", default="")
    p_rb.set_defaults(func=cmd_rollback)

    p_cf = sub.add_parser("classify-failure", help="Classify CI failure and suggest rollback")
    p_cf.add_argument("req_id")
    p_cf.add_argument("--log", default="")
    p_cf.set_defaults(func=cmd_classify_failure)

    p_list = sub.add_parser("list", help="List all workflows")
    p_list.set_defaults(func=cmd_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
