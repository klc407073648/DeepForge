"""Tests for workflow status formatter (REQ-001)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workflow.status_formatter import format_status_summary  # noqa: E402


class TestStatusFormatter(unittest.TestCase):
    def test_AC1_format_includes_req_stage_next(self) -> None:
        state = {
            "req_id": "REQ-001",
            "stage": "PLAN_REVIEW",
            "approvals": {},
        }
        result = format_status_summary(state)
        self.assertIn("REQ: REQ-001", result)
        self.assertIn("Stage: PLAN_REVIEW", result)
        self.assertIn("Next:", result)
        self.assertIn("approve --gate plan", result)

    def test_AC2_shows_plan_approved(self) -> None:
        state = {
            "req_id": "REQ-001",
            "stage": "DIAGRAM_DRAFT",
            "approvals": {
                "plan": {"by": "reviewer", "at": "2026-06-14T00:00:00Z"},
            },
        }
        result = format_status_summary(state)
        self.assertIn("plan: APPROVED", result)
        self.assertIn("reviewer", result)

    def test_AC3_shows_pending_approvals(self) -> None:
        state = {
            "req_id": "REQ-001",
            "stage": "DRAFT_PLAN",
            "approvals": {},
        }
        result = format_status_summary(state)
        self.assertIn("plan: pending", result)
        self.assertIn("diagram: pending", result)
        self.assertIn("no approvals yet", result)


if __name__ == "__main__":
    unittest.main()
