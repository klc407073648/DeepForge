# 概念图确认 Checklist

> 用于 `reviews/{req_id}-diagram-review.md` 审核阶段（stage: DIAGRAM_DRAFT）

## 通过条件

全部必选项勾选后方可 `approve --gate diagram`。

## Checklist

### 必选项

- [ ] 架构图反映方案中的模块与分层
- [ ] 至少一个关键业务流程有时序图
- [ ] 主流程与异常流程均已覆盖
- [ ] 与现有系统边界一致，无方案外模块
- [ ] 数据流向合理（输入 → 处理 → 存储 → 输出）
- [ ] README.md 设计决策与图表一致

### 可选项

- [ ] UI/交互概念图（如有前端需求）
- [ ] 部署/运行时视图（如有运维需求）

## 审核操作

**通过：**

```bash
python scripts/workflow/workflow.py approve REQ-xxx --gate diagram --by <reviewer>
```

**驳回（回到方案修订）：**

```bash
python scripts/workflow/workflow.py reject REQ-xxx --gate diagram --by <reviewer> --reason "具体意见"
```

驳回后使用 `workflow-diagram` skill 修订概念图。
