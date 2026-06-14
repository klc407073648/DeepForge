---
req_id: REQ-001
diagram_version: 1
status: APPROVED
---

# 概念图说明：REQ-001

## 1. 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 格式化逻辑位置 | src/workflow/status_formatter.py | 可独立测试，与 CLI 解耦 |
| NEXT_SKILL 来源 | 从 workflow.py 传入或默认映射 | 避免重复维护 |
| 输出模式 | --human opt-in | 不破坏现有 JSON 管道 |

## 2. 图表索引

| 文件 | 类型 | 说明 |
|------|------|------|
| architecture.mmd | 架构图 | CLI → formatter 数据流 |
| sequence-status.mmd | 时序图 | status --human 调用链 |

## 3. 主流程

用户执行 `status --human` → 加载 state.json → format_status_summary → 打印摘要。

## 4. 异常流程

- state 文件不存在：由 workflow.py 现有逻辑报错退出
- 未知 stage：formatter 显示 stage 原值 + "unknown next step"

## 5. 与现有系统边界

- formatter 不读写文件，仅接收 dict
- workflow.py 负责 I/O 与状态机
