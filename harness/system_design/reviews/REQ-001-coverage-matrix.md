---
req_id: REQ-001
status: PASSED
---

# 测试覆盖矩阵：REQ-001

| AC-ID | 测试文件 | 测试用例名 | 状态 |
|-------|----------|------------|------|
| AC-1 | tests/test_status_formatter.py | test_AC1_format_includes_req_stage_next | pass |
| AC-2 | tests/test_status_formatter.py | test_AC2_shows_plan_approved | pass |
| AC-3 | tests/test_status_formatter.py | test_AC3_shows_pending_approvals | pass |

## 说明

- 每个 AC 至少对应一个测试用例
- 状态：`pending` | `pass` | `fail`
