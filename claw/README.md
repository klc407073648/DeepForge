# Claw

递归爬取公开网页上的需求方案，整理为 Markdown 文档集，并调用 LLM 生成结构化代码生成计划。

```
需求链接 → claw fetch → Markdown 文档集 → claw plan → 代码生成计划
                              └──────────── claw run（一键完成）────────────┘
```

## 安装

```bash
cd claw
pip install -e ".[dev]"
```

安装后会在 Python 的 `Scripts` 目录注册 `claw` 命令。若提示找不到命令，可改用：

```bash
python -m claw.cli --help
```

## 快速开始

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 CLAW_CHAT_API_KEY

# 2. 抓取需求页面
claw fetch https://example.com/spec --depth 2

# 3. 配置目标业务项目上下文（非 claw 自身 README）
cp .claw/context/README.md.example .claw/context/README.md
# 编辑或复制目标项目说明，例如：cp ../portal/README.md .claw/context/README.md

# 4. 生成代码计划（将路径替换为上一步输出的 cache 目录）
claw plan .claw/cache/example.com/20260528T140000Z

# 或一步完成
claw run https://example.com/spec --depth 2

claw run https://example.com/spec --depth 2  --model deepseek-v4-pro
```

---

## 配置

配置分三层，优先级从高到低：**命令行参数 > `.claw.toml` > `.env` 默认值**。

### 环境变量（`.env`）

```bash
cp .env.example .env
```

| 变量 | 说明 | 示例 |
|------|------|------|
| `CLAW_CHAT_API_KEY` | LLM API Key（必填，用于 `plan` / `run`） | `sk-xxx` |
| `CLAW_CHAT_BASE_URL` | OpenAI 兼容接口地址 | `https://api.deepseek.com/v1` |
| `CLAW_CHAT_MODEL` | 默认模型名 | `deepseek-chat` |
| `OPENAI_API_KEY` | 备用 Key（未设置 `CLAW_CHAT_API_KEY` 时生效） | |
| `OPENAI_BASE_URL` | 备用 Base URL | `https://api.openai.com/v1` |
| `CLAW_CACHE_DIR` | 抓取结果根目录 | `.claw/cache` |
| `CLAW_PLANS_DIR` | 计划输出根目录 | `.claw/plans` |

> **注意**：`claw fetch` 不需要 API Key；只有 `claw plan` 和 `claw run` 会调用 LLM。

### LLM 提供商示例

**DeepSeek（推荐）**

```env
CLAW_CHAT_API_KEY=sk-xxx
CLAW_CHAT_BASE_URL=https://api.deepseek.com/v1
CLAW_CHAT_MODEL=deepseek-chat
```

可选模型：`deepseek-chat`（常规）、`deepseek-v4-pro`（推理，更慢更贵）。

**OpenAI**

```env
CLAW_CHAT_API_KEY=sk-xxx
CLAW_CHAT_BASE_URL=https://api.openai.com/v1
CLAW_CHAT_MODEL=gpt-4o
```

**其他 OpenAI 兼容服务**（如本地 Ollama、OneAPI 等）

```env
CLAW_CHAT_API_KEY=your-key
CLAW_CHAT_BASE_URL=http://localhost:11434/v1
CLAW_CHAT_MODEL=qwen2.5
```

### 项目配置（`.claw.toml`，可选）

在项目根目录创建 `.claw.toml`，避免每次传参：

```toml
[crawl]
max_depth = 2              # 最大递归深度（根页面 depth=0）
max_pages = 50             # 最多抓取页数
same_domain_only = true    # 仅跟随同域名链接
max_concurrency = 5        # 并发请求数
request_delay_ms = 200     # 请求间隔（毫秒）
timeout_seconds = 30.0
exclude_patterns = ["*/login*", "*.pdf", "*.zip"]
include_patterns = []      # 非空时仅匹配这些模式的链接会被跟随
ignore_robots = false
no_images = false
max_content_chars = 100000 # 单页 Markdown 最大字符数

[plan]
model = "deepseek-chat"    # 未设置时读取 CLAW_CHAT_MODEL
max_input_chars = 120000   # 送入 LLM 的最大字符数
temperature = 0.2
system_prompt_file = ".claw/prompts/system.md"   # 可选
user_prompt_file = ".claw/prompts/user.md"       # 可选
```

指定其他配置文件：

```bash
claw fetch https://example.com/spec --config ./my-config.toml
```

### 自定义 LLM 提示词

复制示例文件：

```bash
cp .claw/prompts/system.md.example .claw/prompts/system.md
cp .claw/prompts/user.md.example .claw/prompts/user.md
```

在 `.claw.toml` 中引用，或命令行临时指定：

```bash
claw plan .claw/cache/run1 \
  --system-prompt .claw/prompts/system.md \
  --user-prompt .claw/prompts/user.md
```

**user 模板占位符**：

| 占位符 | 含义 |
|--------|------|
| `{{requirements}}` | 聚合后的需求 Markdown |
| `{{repo_context}}` | 仓库上下文纯文本 |
| `{{repo_context_block}}` | 带标题的仓库上下文块 |
| `{{sections}}` | 输出章节列表 |

**优先级**：CLI 参数 > `.claw.toml [plan]` > 内置默认。

### 目标项目上下文（`.claw/context/`）

`claw plan` / `claw run` 需要了解**你要实现需求的目标业务项目**，而不是 claw 工具本身的 README。

推荐在 `.claw/context/` 放置目标项目说明：

```bash
cp .claw/context/README.md.example .claw/context/README.md
# 从目标项目复制，例如：
cp ../portal/README.md .claw/context/README.md
# 可选架构说明
cp .claw/context/architecture.md.example .claw/context/architecture.md
```

未指定 `--repo-context` 时，自动聚合 `.claw/context/*.md`（跳过 `*.example`）。

```toml
# .claw.toml
[plan]
context_dir = ".claw/context"
context_max_chars = 20000
```

显式指定单文件（优先级高于目录）：

```bash
claw plan .claw/cache/run1 --repo-context ../portal/README.md
claw plan .claw/cache/run1 -v   # verbose 显示 context 来源
```

---

## 命令参考

### `claw fetch` — 递归抓取

将需求页面及其子链接抓取为 Markdown 文件。

```bash
claw fetch <URL> [OPTIONS]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `URL` | 需求方案根链接（必填） | — |
| `--out PATH` | 输出目录 | `.claw/cache/<host>/<timestamp>/` |
| `--depth N` | 最大递归深度 | `2` |
| `--max-pages N` | 最多抓取页数 | `50` |
| `--any-domain` | 允许跟随跨域链接 | 关闭（仅同域） |
| `--dry-run` | 只发现链接，不写 Markdown 文件 | 关闭 |
| `--ignore-robots` | 忽略 robots.txt（调试用） | 关闭 |
| `--no-images` | Markdown 中不保留图片 | 关闭 |
| `--min-content-chars N` | 空页过滤阈值（可见文本字符数） | `80` |
| `--config PATH` | 指定 `.claw.toml` 路径 | 自动查找 |
| `-v, --verbose` | 输出详细信息 | 关闭 |

**示例**

```bash
# 基础抓取
claw fetch https://example.com/spec

# 指定深度和输出目录
claw fetch https://example.com/spec --depth 3 --max-pages 100 --out .claw/cache/my-spec

# 预览会抓取哪些页面（不写文件）
claw fetch https://example.com/spec --dry-run -v

# 跨域跟随 + 忽略 robots（谨慎使用）
claw fetch https://example.com/spec --any-domain --ignore-robots

# 精简输出（去掉图片）
claw fetch https://example.com/spec --no-images
```

---

### `claw plan` — 生成代码计划

基于已抓取的 Markdown 文档集，调用 LLM 生成代码生成计划。

```bash
claw plan <CACHE_DIR> [OPTIONS]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CACHE_DIR` | 抓取结果目录（含 `manifest.json` 和 `*.md`） | — |
| `--out PATH` | 计划输出文件路径 | `.claw/plans/<timestamp>-plan.md` |
| `--repo-context PATH` | 目标业务项目上下文单文件（覆盖 `.claw/context/`） | 自动读 `.claw/context/` |
| `--context-dir PATH` | 目标项目上下文目录 | `.claw/context` |
| `--model NAME` | 覆盖模型名 | `.env` / `.claw.toml` 中的配置 |
| `--system-prompt PATH` | 自定义 system prompt 文件 | 内置默认 |
| `--user-prompt PATH` | 自定义 user prompt 模板文件 | 内置默认 |
| `--config PATH` | 指定 `.claw.toml` 路径 | 自动查找 |
| `-v, --verbose` | 输出详细信息（含 context 来源） | 关闭 |

**示例**

```bash
# 基础用法
claw plan .claw/cache/example.com/20260528T140000Z

# 指定输出路径（自动使用 .claw/context/ 作为目标项目上下文）
claw plan .claw/cache/run1 --out ./plans/feature-a-plan.md

# 或显式指定目标项目 README
claw plan .claw/cache/run1 --repo-context ../portal/README.md

# 临时切换模型
claw plan .claw/cache/run1 --model deepseek-reasoner
```

---

### `claw run` — 一键流水线

依次执行 `fetch` + `plan`。

```bash
claw run <URL> [OPTIONS]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `URL` | 需求方案根链接（必填） | — |
| `--depth N` | 最大递归深度 | `2` |
| `--max-pages N` | 最多抓取页数 | `50` |
| `--out PATH` | 抓取结果目录 | `.claw/cache/<host>/<timestamp>/` |
| `--plan-out PATH` | 计划输出文件路径 | `.claw/plans/<timestamp>-plan.md` |
| `--repo-context PATH` | 目标业务项目上下文单文件 | 自动读 `.claw/context/` |
| `--context-dir PATH` | 目标项目上下文目录 | `.claw/context` |
| `--model NAME` | 覆盖模型名 | `.env` / `.claw.toml` 中的配置 |
| `--system-prompt PATH` | 自定义 system prompt 文件 | 内置默认 |
| `--user-prompt PATH` | 自定义 user prompt 模板文件 | 内置默认 |
| `--config PATH` | 指定 `.claw.toml` 路径 | 自动查找 |
| `-v, --verbose` | 输出详细信息 | 关闭 |

**示例**

```bash
# 最简一键流程
claw run https://example.com/spec

# 完整参数
claw run https://example.com/spec \
  --depth 2 \
  --max-pages 30 \
  --repo-context ../portal/README.md \
  --plan-out ./plans/my-plan.md \
  -v
```

---

## 典型使用场景

### 场景 1：分步执行（推荐调试）

适合首次使用或需要检查抓取结果后再生成计划。

```bash
# 第一步：抓取
claw fetch https://docs.example.com/feature-x --depth 2 -v

# 检查 .claw/cache/.../ 下的 Markdown 是否完整

# 第二步：生成计划（默认读取 .claw/context/）
claw plan .claw/cache/docs.example.com/<run-id> \
  --out ./plans/feature-x-plan.md
```

### 场景 2：一键生成计划

适合需求页面结构清晰、配置已调好的情况。

```bash
claw run https://docs.example.com/feature-x --depth 2
```

### 场景 3：大型文档站（限制范围）

```bash
# 限制深度和页数，避免抓取过多
claw fetch https://wiki.example.com/project --depth 1 --max-pages 20

# 或在 .claw.toml 中配置 include_patterns 缩小范围
# include_patterns = ["*/project/feature-x/*"]
```

### 场景 4：仅抓取、暂不调用 LLM

```bash
claw fetch https://example.com/spec --depth 2 --out ./docs/requirements
# 稍后再 plan
claw plan ./docs/requirements
```

### 场景 5：对接 Cursor / 人工 Review

生成计划后，将 `.claw/plans/*.md` 作为 Agent 输入：

```bash
claw run https://example.com/spec
# 打开 .claw/plans/<timestamp>-plan.md 交给 Cursor Agent 执行
```

---

## 输出说明

### 抓取结果（`.claw/cache/`）

```
.claw/cache/<host>/<run-id>/
├── manifest.json     # 页面图结构、元数据、错误列表
├── errors.log        # 失败 URL 记录（如有）
├── index.md          # 根页面（示例文件名，实际按 URL 路径命名）
├── feature-a.md      # 子页面
└── ...
```

**单页 Markdown 格式**（含 YAML front matter）：

```markdown
---
source_url: https://example.com/spec/feature-a
title: "Feature A 需求说明"
depth: 1
parent_url: https://example.com/spec
fetched_at: 2026-05-28T14:00:00Z
---

# Feature A 需求说明

正文内容...
```

**manifest.json 结构**：

```json
{
  "root_url": "https://example.com/spec",
  "created_at": "2026-05-28T14:00:00Z",
  "pages": [
    {
      "path": "spec.md",
      "source_url": "https://example.com/spec",
      "title": "需求总览",
      "depth": 0,
      "parent_url": null,
      "links_to": ["https://example.com/spec/feature-a"],
      "status": "ok"
    }
  ],
  "errors": []
}
```

### 代码生成计划（`.claw/plans/`）

LLM 输出的 Markdown 包含固定 7 个章节：

1. 需求摘要
2. 技术假设与待确认项
3. 模块/文件变更清单（路径 + 职责）
4. 分步实施任务（带依赖顺序）
5. 接口与数据结构草案
6. 测试与验收标准
7. 风险与回滚点

---

## 爬取行为说明

| 行为 | 说明 |
|------|------|
| 调度策略 | BFS（广度优先），便于控制深度 |
| 深度计算 | 根 URL 为 depth=0，每跟随一层 +1 |
| 同域限制 | 默认仅跟随同一 registrable domain 的链接 |
| 去重 | URL 规范化后去 fragment、去重 |
| 自动排除 | 默认跳过 `*.pdf`、`*.zip`、图片、login 等链接 |
| robots.txt | 默认遵守；调试可用 `--ignore-robots` |
| 并发控制 | 默认最多 5 个并发请求，间隔 200ms |
| 容错 | 单页失败不中断，错误写入 `errors.log` 和 `manifest.json` |

---

## 常见问题

### `400 Bad Request`（DeepSeek / 其他兼容 API）

**最常见原因：模型名不匹配。**

确认 `.env` 中模型与服务商一致：

```env
CLAW_CHAT_MODEL=deepseek-chat   # DeepSeek
# 而非 gpt-4o
```

也可用命令行显式指定：

```bash
claw plan .claw/cache/run1 --model deepseek-chat
```

### `401 Unauthorized`

- 检查 `CLAW_CHAT_API_KEY` 是否正确
- 确认 API Key 有余额、未过期

### `Missing API key`

`claw plan` / `claw run` 需要配置 Key。在项目根目录（运行 `claw` 时的当前目录）创建 `.env`。

### 抓取结果为空或不完整

```bash
# 先用 dry-run 查看会抓哪些链接
claw fetch https://example.com/spec --dry-run -v

# 页面可能是 JS 渲染，当前版本仅支持静态 HTML
# 尝试调大深度或检查是否被 robots.txt 拦截
claw fetch https://example.com/spec --depth 3 -v
```

### 计划内容质量不佳

- 在 `.claw/context/` 放置目标业务项目 README，或使用 `--repo-context ../your-project/README.md`
- 检查抓取的 Markdown 是否包含完整需求
- 调大 `[plan] max_input_chars` 或减小 `[crawl] max_pages` 避免无关页面干扰

---

## 步骤输出

默认会打印主流流程步骤（抓取配置、聚合需求、调用 LLM 等）。加 `-v` 可查看完整 LLM 输入：

```bash
claw plan .claw/cache/run1 -v
```

`-v` 额外输出：

- 使用的**模型**、temperature、API Base
- 需求文档 / 仓库上下文的**字符数**与来源
- 完整的 **System 提示词** 和 **User 提示词**（过长时截断显示）

静默模式（仅保留最终结果）：

```bash
claw run https://example.com/spec -q
```

控制台输出会同步写入 `.claw/logs/<timestamp>-<command>.log`（纯文本，无 Rich 样式）。运行结束会提示日志路径。

可通过环境变量 `CLAW_LOGS_DIR` 修改日志目录。

---

## 开发与测试

```bash
# 运行测试
pytest

# 不安装包直接运行
python -m claw.cli fetch https://example.com/spec --dry-run -v
python -m claw.cli plan .claw/cache/run1
```

---

## 设计说明

- **输入**：公开 HTTP(S) 需求方案链接
- **中间产物**：本地 Markdown 文档集 + `manifest.json`
- **输出**：结构化代码生成计划（Markdown）
- **首版不支持**：登录鉴权、JS 渲染页面、飞书/Confluence 专用适配、自动写代码
