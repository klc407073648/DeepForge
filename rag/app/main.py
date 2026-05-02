from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import Settings, settings, MAX_UPLOAD_BYTES
from app.generation.llm import OpenAICompatibleClient
from app.generation.prompts import build_messages
from app.ingestion.pipeline import ingest_documents
from app.models import Citation, HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.retrieval.search import search_relevant_chunks
from app.retrieval.store import get_chroma_client, get_or_create_collection

_chroma_client: Any = None
_collection: Any = None

# chat模型配置
def _chat_api_configured(s: Settings) -> bool:
    return bool(s.resolved_chat_api_key.strip())

# embedding模型配置
def _embedding_configured(s: Settings) -> bool:
    if s.embedding_backend == "local":
        return bool(s.resolved_local_embedding_model_id.strip())
    return bool(s.resolved_embedding_api_key.strip())

# 服务就绪状态检查
def _service_ready_detail(s: Settings) -> tuple[bool, str | None]:
    parts: list[str] = []
    if not _chat_api_configured(s):
        parts.append("chat: set CHAT_API_KEY or OPENAI_API_KEY")
    if not _embedding_configured(s):
        if s.embedding_backend == "local":
            parts.append("embedding: set LOCAL_EMBEDDING_MODEL or EMBEDDING_MODEL")
        else:
            parts.append("embedding: set EMBEDDING_API_KEY or OPENAI_API_KEY")
    if not parts:
        return True, None
    return False, "; ".join(parts)

# embedding要求配置项
def require_embedding_config(s: Settings) -> None:
    if not _embedding_configured(s):
        if s.embedding_backend == "local":
            raise HTTPException(
                status_code=400,
                detail="LOCAL_EMBEDDING_MODEL or EMBEDDING_MODEL is required for local embedding",
            )
        raise HTTPException(
            status_code=400,
            detail="EMBEDDING_API_KEY or OPENAI_API_KEY is required for HTTP embedding",
        )

# query要求配置项
def require_query_config(s: Settings) -> None:
    require_embedding_config(s)
    if not _chat_api_configured(s):
        raise HTTPException(
            status_code=400,
            detail="CHAT_API_KEY or OPENAI_API_KEY is required for chat",
        )

# @asynccontextmanager 是 Python contextlib 模块提供的一个装饰器，
# 它的核心作用是将一个异步生成器函数（async def 函数中包含 yield）转变为一个异步上下文管理器，
# 从而可以优雅地与 async with 语句配合使用。

# 在 yield 之前做启动逻辑，yield 之后做关闭逻辑。
# 函数功能：获取单例配置settings，并将向量库做一次初始化
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chroma_client, _collection
    s = settings
    _chroma_client = get_chroma_client(s.chroma_persist_dir)
    _collection = get_or_create_collection(_chroma_client, s.collection_name)
    yield
    _collection = None
    _chroma_client = None

# 获取相关配置
def get_settings() -> Settings:
    return settings


def get_collection() -> Any:
    if _collection is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return _collection


def get_llm_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(settings)
# 获取相关配置

# 主程序
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# 健康度检查
@app.get("/health", response_model=HealthResponse)
async def health(s: Settings = Depends(get_settings)) -> HealthResponse:
    ready, detail = _service_ready_detail(s)
    return HealthResponse(
        status="ok",
        app=s.app_name,
        version=s.app_version,
        ready=ready,
        detail=detail,
    )

# 上传内容存储
async def _save_uploads_to_temp(
    uploads: list[UploadFile],
    raw_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for uf in uploads:
        if not uf.filename:
            continue
        data = await uf.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File {uf.filename} exceeds {MAX_UPLOAD_BYTES} bytes",
            )
        safe = Path(uf.filename).name
        dest = raw_dir / safe
        dest.write_bytes(data)
        paths.append(dest)
    return paths

# 导入
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: Annotated[
        UploadFile,
        File(description="单个文档：.txt / .md / .pdf（Swagger 里应显示为「选择文件」）"),
    ],
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> IngestResponse:
    """单文件上传；在 Swagger 中可正确选择本地文件。多文件请用 ``POST /ingest/batch`` 或多次调用本接口。"""
    require_embedding_config(s)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    try:
        paths = await _save_uploads_to_temp([file], raw_dir)
        if not paths:
            raise HTTPException(status_code=400, detail="No valid file uploaded")

        n, sources, elapsed = await ingest_documents(paths, collection, client, s)
        return IngestResponse(indexed_chunks=n, sources=sources, seconds=round(elapsed, 3))
    finally:
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

# 批量导入
@app.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(
    files: Annotated[
        list[UploadFile],
        File(description="同一字段名重复添加多个文件；若界面仍异常，请用 curl 或单文件接口 /ingest"),
    ],
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> IngestResponse:
    """多文件上传（部分 Swagger 版本对数组文件展示不佳，优先用 ``/ingest`` 或 curl）。"""
    require_embedding_config(s)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    try:
        paths = await _save_uploads_to_temp(files, raw_dir)
        if not paths:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        n, sources, elapsed = await ingest_documents(paths, collection, client, s)
        return IngestResponse(indexed_chunks=n, sources=sources, seconds=round(elapsed, 3))
    finally:
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

# 知识检索
@app.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> QueryResponse:
    require_query_config(s)

    chunks = await search_relevant_chunks(body.question, collection, client, s)

    if not chunks:
        return QueryResponse(
            answer="知识库中暂无相关内容，请先上传并索引文档。",
            citations=[],
            no_relevant_context=True,
        )

    best = min(chunks, key=lambda c: c.distance)
    if s.max_retrieval_distance is not None and best.distance > s.max_retrieval_distance:
        return QueryResponse(
            answer="在知识库中未找到与问题足够相关的可靠片段。",
            citations=[],
            no_relevant_context=True,
        )

    messages, mapping = build_messages(body.question, chunks, s)
    answer = await client.chat_completion(messages)

    citations = [
        Citation(
            id=cid,
            text=ch.text[:2000] + ("…" if len(ch.text) > 2000 else ""),
            source=ch.source,
            distance=ch.distance,
        )
        for cid, ch in mapping
    ]

    return QueryResponse(answer=answer, citations=citations, no_relevant_context=False)
