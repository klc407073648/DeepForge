---
req_id: REQ-001
plan_version: 1
status: APPROVED
title: "工作流状态人性化输出"
created_at: "2026-06-14T00:00:00Z"
updated_at: "2026-06-14T00:00:00Z"
---

# 实现方案：工作流状态人性化输出

> 关联需求：[requirements/REQ-001.md](../requirements/REQ-001.md)

## 1. 实现范围

### 做

- 新增 `src/workflow/status_formatter.py`
- 修改 `scripts/workflow/workflow.py` 增加 `--human`  flag
- 新增 `tests/test_status_formatter.py`

### 不做

- 修改状态机逻辑
- Web 界面

## 2. 影响文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| src/workflow/status_formatter.py | 新增 | format_status_summary 纯函数 |
| src/workflow/__init__.py | 新增 | 包初始化 |
| scripts/workflow/workflow.py | 修改 | status 命令支持 --human |
| tests/test_status_formatter.py | 新增 | AC-1~3 测试 |

## 3. 接口 / API 变更

```python
def format_status_summary(state: dict, next_skill_map: dict[str, str] | None = None) -> str: ...
```

CLI: `python scripts/workflow/workflow.py status REQ-001 --human`

## 4. 数据模型变更

无

## 5. 实现步骤

1. 创建 status_formatter 模块
2. 编写单元测试（TDD）
3. 集成到 workflow.py status 命令
4. 跑 CI 验证

## 6. 测试策略

| 类型 | 覆盖范围 | 对应 AC |
|------|----------|---------|
| 单元测试 | format_status_summary 各分支 | AC-1, AC-2, AC-3 |
| 集成测试 | 无（纯函数） | — |

## 7. 风险与回滚

| 风险 | 回滚方案 |
|------|----------|
| --human 破坏 JSON 输出 | 默认仍输出 JSON，--human 为 opt-in |

## 8. 预估工作量

| 阶段 | 预估 |
|------|------|
| 实现 | 2h |
| 测试 | 1h |
| 合计 | 3h |
