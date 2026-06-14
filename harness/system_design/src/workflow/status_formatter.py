"""Human-readable workflow status formatting."""

from __future__ import annotations

from typing import Any

DEFAULT_NEXT_SKILL: dict[str, str] = {
    "DRAFT_PLAN": "workflow-requirement → workflow-architect",
    "PLAN_REVIEW": "人工审核方案 → approve --gate plan",
    "DIAGRAM_DRAFT": "workflow-diagram → approve --gate diagram",
    "DIAGRAM_APPROVED": "workflow-implement",
    "CODE_GEN": "workflow-implement (继续) 或 advance --to TEST_GEN",
    "TEST_GEN": "workflow-tdd",
    "CI_RUNNING": "等待 CI 或 run_ci.py",
    "PASSED": "创建 PR，流程完成",
    "FAILED": "workflow-fix → classify-failure → rollback",
}


def format_status_summary(
    state: dict[str, Any],
    next_skill_map: dict[str, str] | None = None,
) -> str:
    """Format workflow state as a human-readable multi-line summary."""
    skill_map = next_skill_map or DEFAULT_NEXT_SKILL
    req_id = state.get("req_id", "UNKNOWN")
    stage = state.get("stage", "UNKNOWN")
    next_step = skill_map.get(stage, "unknown next step")

    lines = [
        f"REQ: {req_id}",
        f"Stage: {stage}",
        f"Next: {next_step}",
        "",
        "Approvals:",
    ]

    approvals: dict[str, Any] = state.get("approvals") or {}
    for gate in ("plan", "diagram", "code"):
        if gate in approvals:
            info = approvals[gate]
            by = info.get("by", "?")
            at = info.get("at", "?")
            lines.append(f"  - {gate}: APPROVED (by {by} at {at})")
        else:
            lines.append(f"  - {gate}: pending")

    if not approvals:
        lines.append("  (no approvals yet — complete plan and diagram review)")

    plan_ver = state.get("plan_version")
    if plan_ver is not None:
        lines.extend(["", f"Plan version: {plan_ver}"])

    failure_count = state.get("failure_count", 0)
    if failure_count:
        lines.append(f"Failure count: {failure_count}")

    return "\n".join(lines)
