"""Unit tests for workflow orchestrator."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

WORKFLOW_DIR = Path(__file__).resolve().parent
SCRIPTS_WORKFLOW = WORKFLOW_DIR.parent
sys.path.insert(0, str(SCRIPTS_WORKFLOW))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "workflow_orchestrator", SCRIPTS_WORKFLOW / "workflow.py"
)
assert _spec and _spec.loader
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)


class TestWorkflowOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        self.workflows = self.root / ".workflows"
        self.templates = self.root / "templates"
        self.templates.mkdir()
        (self.templates / "state.json").write_text(
            json.dumps(
                {
                    "req_id": "REQ-XXX",
                    "stage": "DRAFT_PLAN",
                    "plan_version": 1,
                    "diagram_version": 1,
                    "source_url": "",
                    "failure_count": 0,
                    "max_failures_before_escalate": 3,
                    "approvals": {},
                    "history": [],
                }
            ),
            encoding="utf-8",
        )
        for name in ("requirement.md", "plan.md", "review.md", "diagram-readme.md"):
            (self.templates / name).write_text(f"# {name}\nreq: REQ-XXX\n", encoding="utf-8")

        self.patches = [
            patch.object(wf, "ROOT", self.root),
            patch.object(wf, "WORKFLOWS", self.workflows),
            patch.object(wf, "TEMPLATES", self.templates),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def test_init_creates_state(self) -> None:
        args = type("Args", (), {"req_id": "REQ-099", "url": "", "title": "T", "by": "test", "force": False})()
        wf.cmd_init(args)
        state = json.loads((self.workflows / "REQ-099" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["stage"], "DRAFT_PLAN")
        self.assertEqual(state["req_id"], "REQ-099")

    def test_approve_plan_advances_stage(self) -> None:
        wf.WORKFLOWS.mkdir(parents=True, exist_ok=True)
        req = "REQ-100"
        state = {
            "req_id": req,
            "stage": "PLAN_REVIEW",
            "plan_version": 1,
            "approvals": {},
            "history": [],
        }
        (self.workflows / req).mkdir()
        (self.workflows / req / "state.json").write_text(json.dumps(state), encoding="utf-8")

        args = type("Args", (), {"req_id": req, "gate": "plan", "by": "reviewer"})()
        wf.cmd_approve(args)
        updated = json.loads((self.workflows / req / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(updated["stage"], "DIAGRAM_DRAFT")
        self.assertIn("plan", updated["approvals"])

    def test_validate_implement_blocked_without_approvals(self) -> None:
        req = "REQ-101"
        (self.workflows / req).mkdir(parents=True)
        state = {"req_id": req, "stage": "DIAGRAM_APPROVED", "approvals": {}}
        (self.workflows / req / "state.json").write_text(json.dumps(state), encoding="utf-8")

        args = type("Args", (), {"req_id": req, "action": "implement"})()
        with self.assertRaises(SystemExit):
            wf.cmd_validate(args)

    def test_classify_failure_test_error(self) -> None:
        log = "FAILED tests/test_foo.py - AssertionError: expected 1 but got 2"
        result = wf.classify_log(log)
        self.assertEqual(result["rollback_stage"], "TEST_GEN")


if __name__ == "__main__":
    unittest.main()
