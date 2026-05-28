"""Prompt templates for code plan generation."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = """你是一位资深软件架构师。根据用户提供的需求文档，输出一份可执行的「代码生成计划」。

要求：
- 不要输出完整实现代码，只输出计划、结构与关键接口草案
- 计划应可直接交给开发 Agent 分步执行
- 使用 Markdown 格式，章节标题必须与下方结构一致
- 使用简体中文
"""

DEFAULT_SECTIONS = """
1. 需求摘要
2. 技术假设与待确认项
3. 模块/文件变更清单（路径 + 职责）
4. 分步实施任务（带依赖顺序）
5. 接口与数据结构草案
6. 测试与验收标准
7. 风险与回滚点
""".strip()

DEFAULT_USER_PROMPT_TEMPLATE = """请根据以下需求文档，生成代码生成计划。

输出必须包含以下章节：
{{sections}}

---

## 需求文档

{{requirements}}
{{repo_context_block}}
"""


def load_prompt_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def format_sections(sections: list[str] | None) -> str:
    if sections:
        return "\n".join(sections)
    return DEFAULT_SECTIONS


def render_template(template: str, variables: dict[str, str]) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def build_repo_context_block(repo_context: str | None) -> str:
    if not repo_context:
        return ""
    return f"\n---\n\n## 目标仓库上下文\n\n{repo_context}\n"


def build_user_prompt(
    requirements_md: str,
    repo_context: str | None = None,
    *,
    template: str | None = None,
    sections: list[str] | None = None,
) -> str:
    tpl = template or DEFAULT_USER_PROMPT_TEMPLATE
    return render_template(
        tpl,
        {
            "requirements": requirements_md,
            "repo_context": repo_context or "",
            "repo_context_block": build_repo_context_block(repo_context),
            "sections": format_sections(sections),
        },
    )


def resolve_system_prompt(
    *,
    cli_path: str | None = None,
    config_path: str | None = None,
) -> str:
    for path in (cli_path, config_path):
        content = load_prompt_file(path)
        if content:
            return content
    return DEFAULT_SYSTEM_PROMPT


def resolve_user_prompt_template(
    *,
    cli_path: str | None = None,
    config_path: str | None = None,
) -> str | None:
    for path in (cli_path, config_path):
        content = load_prompt_file(path)
        if content:
            return content
    return None
