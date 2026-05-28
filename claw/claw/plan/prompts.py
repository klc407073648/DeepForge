"""Prompt templates for code plan generation."""

SYSTEM_PROMPT = """你是一位资深软件架构师。根据用户提供的需求文档，输出一份可执行的「代码生成计划」。

要求：
- 不要输出完整实现代码，只输出计划、结构与关键接口草案
- 计划应可直接交给开发 Agent 分步执行
- 使用 Markdown 格式，章节标题必须与下方结构一致
- 使用简体中文
"""

PLAN_SECTIONS = """
1. 需求摘要
2. 技术假设与待确认项
3. 模块/文件变更清单（路径 + 职责）
4. 分步实施任务（带依赖顺序）
5. 接口与数据结构草案
6. 测试与验收标准
7. 风险与回滚点
"""


def build_user_prompt(requirements_md: str, repo_context: str | None = None) -> str:
    parts = [
        "请根据以下需求文档，生成代码生成计划。",
        "",
        "输出必须包含以下章节：",
        PLAN_SECTIONS.strip(),
        "",
        "---",
        "",
        "## 需求文档",
        "",
        requirements_md,
    ]
    if repo_context:
        parts.extend(["", "---", "", "## 目标仓库上下文", "", repo_context])
    return "\n".join(parts)
