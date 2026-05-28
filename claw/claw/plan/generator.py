"""Generate code plan via LLM."""

from __future__ import annotations

from pathlib import Path

from claw.config import PlanConfig, Settings
from claw.crawl.rules import SiteRule, apply_site_rule_to_plan
from claw.llm.client import ChatClient
from claw.plan.aggregator import aggregate_requirements, load_repo_context
from claw.plan.prompts import build_user_prompt, resolve_system_prompt, resolve_user_prompt_template


async def generate_plan(
    cache_dir: Path,
    output_path: Path,
    *,
    settings: Settings,
    plan_config: PlanConfig,
    repo_context_path: Path | None = None,
    site_rule: SiteRule | None = None,
    system_prompt_path: str | None = None,
    user_prompt_path: str | None = None,
) -> Path:
    if not settings.resolved_chat_api_key:
        raise ValueError(
            "Missing API key. Set CLAW_CHAT_API_KEY or OPENAI_API_KEY in .env"
        )

    apply_site_rule_to_plan(plan_config, site_rule)

    requirements = aggregate_requirements(cache_dir, plan_config.max_input_chars)
    repo_context = load_repo_context(repo_context_path)

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

    client = ChatClient(settings)
    plan_text = await client.chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=plan_config.model,
        temperature=plan_config.temperature,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan_text + "\n", encoding="utf-8")
    return output_path
