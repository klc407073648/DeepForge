---
name: closed-loop-workflow
description: Orchestrates the full closed-loop development pipeline from requirement URL to plan, diagram, code, tests, and CI. Use when starting a new REQ, continuing workflow stages, or asking how the development pipeline works.
---

# 闭环开发工作流（总览）

> 完整文档见 [WORKFLOW.md](../../WORKFLOW.md)（系统概述、快速入门、完整流程、Cursor 集成、FAQ）

## 快速开始

> 所有命令在 `system_design/` 目录下执行。详见 [WORKFLOW.md §2](../../WORKFLOW.md#2-快速入门)。

```bash
# 1. 初始化
python scripts/workflow/workflow.py init REQ-002 --url "https://..." --title "功能名"

# 2. 查看状态
python scripts/workflow/workflow.py status REQ-002 --human

# 3. 各阶段使用对应 Skill（见下表）

# 4. 审核通过
python scripts/workflow/workflow.py approve REQ-002 --gate plan --by user
python scripts/workflow/workflow.py approve REQ-002 --gate diagram --by user

# 5. CI 验证
python scripts/workflow/run_ci.py REQ-002
python scripts/workflow/workflow.py approve REQ-002 --gate code --by user
```

## 阶段 → Skill 映射

| Stage | Skill | Checklist |
|-------|-------|-----------|
| DRAFT_PLAN | workflow-requirement, workflow-architect | — |
| PLAN_REVIEW | — | checklists/plan-review.md |
| DIAGRAM_DRAFT | workflow-diagram | checklists/diagram-review.md |
| DIAGRAM_APPROVED+ | workflow-implement | — |
| TEST_GEN | workflow-tdd | checklists/code-delivery.md |
| FAILED | workflow-fix | — |

## 试点参考

完整端到端示例见 `REQ-001`：

- [requirements/REQ-001.md](../../requirements/REQ-001.md)
- [plans/REQ-001-plan-v1.md](../../plans/REQ-001-plan-v1.md)
- [diagrams/REQ-001/](../../diagrams/REQ-001/)
- [.workflows/REQ-001/state.json](../../.workflows/REQ-001/state.json)

## 产物目录

```text
.workflows/{req_id}/state.json
requirements/{req_id}.md
plans/{req_id}-plan-vN.md
diagrams/{req_id}/
reviews/{req_id}-*-review.md
templates/          # 模板
checklists/         # 审核清单
```

## 硬规则

1. 未 approve plan + diagram 前禁止写业务代码
2. 写代码前：`python scripts/workflow/workflow.py validate REQ-xxx --action implement`
3. CI 失败：`classify-failure` → `rollback` → `workflow-fix`
