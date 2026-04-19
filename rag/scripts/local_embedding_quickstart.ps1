# 在仓库 rag 目录下执行:  .\scripts\local_embedding_quickstart.ps1
# 先安装依赖: pip install -r requirements.txt
# 首次跑本地模型会从 Hugging Face 下载权重，需网络；可设环境变量 HF_HOME 指定缓存目录。

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$root = Get-Location

# ---- 按你的环境修改以下变量 ----
# 对话仍走 OpenAI 兼容 API（如 DeepSeek / OpenAI），与本地嵌入互不冲突
$env:OPENAI_API_KEY = "YOUR_CHAT_API_KEY"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:CHAT_MODEL = "deepseek-chat"

# 本地嵌入（SentenceTransformers）
$env:EMBEDDING_BACKEND = "local"
$env:LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 可选: cpu | cuda ，留空则由 sentence-transformers 自选
# $env:LOCAL_EMBEDDING_DEVICE = "cpu"
# $env:LOCAL_EMBEDDING_BATCH_SIZE = "32"

# 若索引与对话曾用别的嵌入维度，请换新集合名或清空 chroma 后再 ingest
# $env:COLLECTION_NAME = "rag_docs_local_bge"

Write-Host "Working directory: $root"
Write-Host "Starting uvicorn..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
