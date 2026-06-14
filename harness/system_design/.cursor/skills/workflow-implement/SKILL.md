---
name: workflow-implement
description: Implements business code from approved plans and diagrams in batches. Use only after diagram approval when writing production code for a REQ workflow, following the file list in the plan.
---

# 实现 Agent

## 前置条件（硬门禁）

- state.stage 必须为 `DIAGRAM_APPROVED`、`CODE_GEN` 或 `TEST_GEN`
- state.approvals.plan 和 state.approvals.diagram 必须存在
- 若门禁不满足，**立即停止**并提示用户先完成方案与概念图审核

验证命令：

```bash
python scripts/workflow/workflow.py validate {req_id} --action implement
```

## 约束

- 仅修改方案「影响文件清单」中的文件
- 不引入方案未批准的新依赖
- 不跳过鉴权、校验、错误处理
- 遵循 `AGENTS.md` 编码规范
- 每批 3–5 个文件，每批后运行 lint/相关测试

## 工作流

1. 阅读 `plans/{req_id}-plan-v{N}.md`、`diagrams/{req_id}/`、相关现有代码
2. 创建特性分支：`feat/{req_id}-*`
3. 按实现步骤分批修改代码
4. 更新 state：`stage` → `CODE_GEN`
5. 每批完成后记录 progress 到 state.history

## Guardrails

- 方案外文件：**不修改**
- 公共 API：**不随意变更**
- 大规模格式化：**禁止**

## 完成后

更新 state.stage → `TEST_GEN`，提示使用 `workflow-tdd` skill 生成测试。

```bash
python scripts/workflow/workflow.py advance {req_id} --to TEST_GEN
```
