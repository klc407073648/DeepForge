---
name: workflow-fix
description: Analyzes CI failures and produces minimal fixes with rollback_stage classification. Use when CI fails in the closed-loop workflow and the agent needs to determine whether to fix code, tests, plan, or diagrams.
---

# 修复 Agent

## 前置条件

- state.stage 为 `CI_RUNNING` 或 `FAILED`
- 有 CI 日志（`.workflows/{req_id}/ci-last.log` 或终端输出）

## 约束

- 最小化修复，不做无关重构
- 必须输出 `root_cause` 和 `rollback_stage`
- 同一 REQ 失败超过 max_failures_before_escalate 次时 **停止自动修复**， escalate 人工

## 失败分类

| 类型 | rollback_stage | 动作 |
|------|----------------|------|
| 测试断言/用例错误 | TEST_GEN | 修正测试 |
| 业务逻辑错误 | CODE_GEN | 修正代码 |
| 方案遗漏/范围不足 | PLAN_REVIEW | 修订方案，可能需重画概念图 |
| 架构/模块划分错误 | DIAGRAM_DRAFT | 修订概念图后重新实现 |

分类命令：

```bash
python scripts/workflow/workflow.py classify-failure {req_id} --log .workflows/{req_id}/ci-last.log
```

## 工作流

1. 阅读 CI 日志，定位首个失败测试/错误
2. 判断 root_cause 与 rollback_stage
3. 写入 `reviews/{req_id}-fix-{timestamp}.md`：
   - root_cause
   - rollback_stage
   - fix_summary
4. 执行 orchestrator 回退：

```bash
python scripts/workflow/workflow.py rollback {req_id} --to {rollback_stage} --reason "..."
```

5. 在目标阶段做最小修复
6. 重新运行 CI

## 输出模板

```markdown
## root_cause
...

## rollback_stage
CODE_GEN | TEST_GEN | PLAN_REVIEW | DIAGRAM_DRAFT

## fix_summary
...
```
