# 代码交付 Checklist

> 用于 CI 通过后最终交付确认（stage: CI_RUNNING → PASSED）

## 通过条件

全部必选项勾选后方可 `approve --gate code` 或合并 PR。

## Checklist

### 必选项

- [ ] 所有 AC 在 `reviews/{req_id}-coverage-matrix.md` 中有对应测试
- [ ] CI 全绿（lint + test）
- [ ] 无方案外文件改动
- [ ] 无未批准的新依赖
- [ ] 鉴权/校验/错误处理未跳过
- [ ] 遵循 AGENTS.md 编码与测试规范

### 可选项

- [ ] PR 描述含：变更背景、测试结果、风险、回滚方案
- [ ] 增量测试覆盖率满足团队门禁

## 审核操作

**CI 本地验证：**

```bash
python scripts/workflow/run_ci.py REQ-xxx
```

**交付确认：**

```bash
python scripts/workflow/workflow.py approve REQ-xxx --gate code --by <reviewer>
```

## CI 失败处理

```bash
python scripts/workflow/workflow.py classify-failure REQ-xxx
python scripts/workflow/workflow.py rollback REQ-xxx --to <rollback_stage> --reason "..."
```

使用 `workflow-fix` skill 做最小修复后重新跑 CI。
