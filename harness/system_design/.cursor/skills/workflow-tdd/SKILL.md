---
name: workflow-tdd
description: Generates unit and integration tests mapped 1:1 to acceptance criteria AC-x. Use after code implementation in the closed-loop workflow, or when creating test coverage for a REQ feature.
---

# TDD Agent

## 前置条件

- `requirements/{req_id}.md` 含结构化 acceptance_criteria
- 业务代码已实现（state.stage 为 `TEST_GEN` 或 `CODE_GEN`）
- 读取 `diagrams/{req_id}/` 确定集成测试边界

## 约束

- 每个 AC-ID 至少一个测试用例，命名含 AC ID（如 `test_AC1_...`）
- 测试遵循项目现有测试框架与目录约定（见 AGENTS.md）
- **禁止**修改业务逻辑除非测试暴露明确缺陷（应交给 workflow-fix）

## 工作流

1. 从需求 yaml 提取所有 AC
2. 生成测试骨架（Red）
3. 对照已实现代码补全断言（Green）
4. 编写 `reviews/{req_id}-coverage-matrix.md`（基于 [templates/test-coverage-matrix.md](../../templates/test-coverage-matrix.md)）
5. 运行测试：

```bash
python scripts/workflow/run_ci.py {req_id}
```

6. 更新 state：`stage` → `CI_RUNNING`

## 测试分层

| 类型 | 来源 | 位置 |
|------|------|------|
| 单元测试 | AC-x | tests/unit/ |
| 集成测试 | 概念图模块交互 | tests/integration/ |

## 完成后

- CI 通过：state → `PASSED`
- CI 失败：使用 `workflow-fix` skill，或 `python scripts/workflow/workflow.py classify-failure {req_id}`
