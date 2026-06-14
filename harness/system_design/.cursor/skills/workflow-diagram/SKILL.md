---
name: workflow-diagram
description: Generates Mermaid concept diagrams from approved plans into diagrams/REQ-xxx/. Use after plan approval when creating architecture, sequence, or data flow diagrams before code implementation.
---

# 概念图 Agent

## 前置条件

- `plans/{req_id}-plan-v{N}.md` 存在且 state.approvals.plan 已记录
- state.stage 为 `DIAGRAM_DRAFT` 或 `PLAN_REVIEW` 且 plan 已 APPROVED
- 运行 approve 后 orchestrator 会将 stage 设为 `DIAGRAM_DRAFT`

## 约束

- **禁止**生成业务代码或测试代码
- 概念图必须与已批准方案一致，不得引入方案外模块
- 输出目录：`diagrams/{req_id}/`
- 必须包含 `README.md`（基于 [templates/diagram-readme.md](../../templates/diagram-readme.md)）

## 推荐产出

| 场景 | 文件 | Mermaid 类型 |
|------|------|--------------|
| 模块关系 | architecture.mmd | flowchart |
| 业务流程 | sequence-*.mmd | sequenceDiagram |
| 数据流 | dataflow.mmd | flowchart |

## Mermaid 规范

- 节点 ID 用 camelCase，不用空格
- 边标签含特殊字符时用双引号
- 不用 style/color 着色

## 工作流

1. 阅读已批准方案与需求验收标准
2. 生成架构图 + 至少一个关键时序图
3. 编写 `diagrams/{req_id}/README.md` 说明设计决策
4. 更新 state：`stage` → `DIAGRAM_DRAFT`（待确认）
5. 创建 `reviews/{req_id}-diagram-review.md`

## 完成后

提示用户使用 [checklists/diagram-review.md](../../checklists/diagram-review.md) 确认，通过后：

```bash
python scripts/workflow/workflow.py approve {req_id} --gate diagram --by <reviewer>
```
