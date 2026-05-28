"""Generate code plan via LLM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from claw.config import PlanConfig, Settings
from claw.llm.client import ChatClient
from claw.plan.aggregator import aggregate_requirements, resolve_repo_context
from claw.plan.prompts import (
    build_user_prompt,
    resolve_system_prompt,
    resolve_user_prompt_template,
)
from claw.progress import PlanInputSummary

if TYPE_CHECKING:
    from claw.progress import StepLogger


async def generate_plan(
    cache_dir: Path,
    output_path: Path,
    *,
    settings: Settings,
    plan_config: PlanConfig,
    repo_context_path: Path | None = None,
    system_prompt_path: str | None = None,
    user_prompt_path: str | None = None,
    logger: StepLogger | None = None,
) -> Path:
    if not settings.resolved_chat_api_key:
        raise ValueError(
            "Missing API key. Set CLAW_CHAT_API_KEY or OPENAI_API_KEY in .env"
        )

    if logger:
        logger.step("加载并聚合需求文档")
        logger.info(f"目录: {cache_dir}")

    requirements = aggregate_requirements(cache_dir, plan_config.max_input_chars)

    if logger:
        logger.info(f"需求文档: {len(requirements):,} 字符")
        logger.step("加载目标项目上下文")

    repo_context, context_source = resolve_repo_context(
        repo_context_path,
        plan_config.context_dir,
        max_chars=plan_config.context_max_chars,
    )

    if logger:
        ctx_len = len(repo_context) if repo_context else 0
        logger.info(f"来源: {context_source}，{ctx_len:,} 字符")
        logger.step("构建 LLM 提示词")

    system_prompt = resolve_system_prompt(
        cli_path=system_prompt_path,
        config_path=plan_config.system_prompt_file,
    )
    user_template = resolve_user_prompt_template(
        cli_path=user_prompt_path,
        config_path=plan_config.user_prompt_file,
    )
    user_prompt = build_user_prompt(
        requirements,
        repo_context,
        template=user_template,
        sections=plan_config.sections or None,
    )

    if logger:
        logger.print_plan_inputs(
            PlanInputSummary(
                model=plan_config.model,
                temperature=plan_config.temperature,
                api_base=settings.resolved_chat_base_url,
                cache_dir=str(cache_dir),
                output_path=str(output_path),
                context_source=context_source,
                requirements_chars=len(requirements),
                repo_context_chars=len(repo_context or ""),
                system_prompt_chars=len(system_prompt),
                user_prompt_chars=len(user_prompt),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )
        logger.step(f"调用 LLM 生成计划（模型: {plan_config.model}）")

    client = ChatClient(settings)
    plan_text = await client.chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=plan_config.model,
        temperature=plan_config.temperature,
    )

    if logger:
        logger.info(f"LLM 返回: {len(plan_text):,} 字符")
        logger.step("写入计划文件")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan_text + "\n", encoding="utf-8")
    return output_path
