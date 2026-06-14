# 方案审核 Checklist

> 用于 `reviews/{req_id}-plan-review.md` 审核阶段（stage: PLAN_REVIEW）

## 通过条件

全部必选项勾选后方可 `approve --gate plan`。

## Checklist

### 必选项

- [ ] 验收标准完整且每条 AC 可测试（Given-When-Then）
- [ ] 范围边界清晰，「不做」列表明确
- [ ] 影响文件清单完整，无遗漏关键模块
- [ ] 接口/API 变更已说明（或明确写「无」）
- [ ] 测试策略覆盖所有 AC
- [ ] 风险与回滚方案已评估
- [ ] 无未批准的新依赖或数据库变更
- [ ] 方案未超出需求文档范围

### 可选项

- [ ] 性能/安全非功能需求有对应设计
- [ ] 与现有架构/抽象一致，无重复造轮子
- [ ] 工作量预估合理

## 审核操作

**通过：**

```bash
python scripts/workflow/workflow.py approve REQ-xxx --gate plan --by <reviewer>
```

**驳回：**

```bash
python scripts/workflow/workflow.py reject REQ-xxx --gate plan --by <reviewer> --reason "具体意见"
```

驳回后使用 `workflow-architect` skill 修订，生成 `plans/REQ-xxx-plan-vN+1.md`。
