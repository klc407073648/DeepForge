---
name: workflow-requirement
description: Parses requirement document links or text into structured requirements/REQ-xxx.md with acceptance criteria. Use when starting a closed-loop workflow, parsing a requirement URL, or when the user asks to extract requirements from a document.
---

# 需求解析 Agent

## 前置条件

- 读取 `.workflows/{req_id}/state.json`，确认 `stage` 为 `DRAFT_PLAN` 或正在初始化
- 若 state 不存在，先运行 `python scripts/workflow/workflow.py init {req_id}`

## 约束

- **禁止**生成方案、概念图或代码
- 输出必须写入 `requirements/{req_id}.md`，基于 [templates/requirement.md](../../templates/requirement.md)
- 每条验收标准必须有唯一 ID（AC-1, AC-2…）且 Given-When-Then 可测试

## 工作流

1. 获取需求来源：URL、粘贴正文、或本地文件
2. 若为 URL，抓取正文（处理鉴权失败时提示用户提供正文）
3. 提取：背景、用户故事、验收标准、范围（做/不做）、非功能需求、依赖、风险
4. 写入 `requirements/{req_id}.md`，更新 frontmatter 中 `status: DRAFT`
5. 更新 state：`stage` 保持 `DRAFT_PLAN`，记录 history

## 输出格式

YAML frontmatter 必填字段：

```yaml
req_id: REQ-XXX
title: "简短标题"
status: DRAFT
source_url: "原始链接"
created_at: ISO8601
updated_at: ISO8601
```

验收标准块必须使用 plan 规定的 yaml 结构。

## 完成后

提示用户：「需求已结构化，请使用 `workflow-architect` skill 生成方案，或运行 `python scripts/workflow/workflow.py next {req_id}` 查看下一步。」
