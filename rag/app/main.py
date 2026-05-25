from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, settings, MAX_UPLOAD_BYTES
from app.documents import delete_document, get_document_detail, list_indexed_documents
from app.generation.llm import OpenAICompatibleClient
from app.generation.prompts import build_messages
from app.logging_config import configure_logging, get_logger
from app.ingestion.pipeline import ingest_documents
from app.models import (
    ChunkingSettings,
    Citation,
    CollectionInfo,
    CollectionsResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    KeyConfigItem,
    KeyConfigListResponse,
    KeyConfigUpsertRequest,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
)
from app.retrieval.search import search_relevant_chunks
from app.retrieval.store import get_chroma_client, get_or_create_collection
from app.runtime_config import (
    delete_key_config,
    get_chunking_settings,
    get_effective_settings,
    list_available_models,
    list_key_configs,
    update_chunking_settings,
    upsert_key_config,
)

log = get_logger(__name__)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chroma_client, _collection
    s = get_effective_settings()
    configure_logging(s.log_level, s.resolved_log_file)
    log.info(
        "Logging ready: level=%s, file=%s",
        s.log_level,
        str(s.resolved_log_file) if s.resolved_log_file else "(console only)",
    )
    _chroma_client = get_chroma_client(s.chroma_persist_dir)
    _collection = get_or_create_collection(_chroma_client, s.collection_name)
    log.debug("Chroma ready: persist_dir=%s collection=%s", s.chroma_persist_dir, s.collection_name)
    yield
    log.info("Shutting down: closing vector store handles")
    _collection = None
    _chroma_client = None

def get_settings() -> Settings:
    return get_effective_settings()


def get_collection() -> Any:
    if _collection is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return _collection


def get_collection_by_name(name: str) -> Any:
    if _chroma_client is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return get_or_create_collection(_chroma_client, name)


def get_llm_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(get_effective_settings())

app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health(s: Settings = Depends(get_settings)) -> HealthResponse:
    ready, detail = _service_ready_detail(s)
    log.debug("GET /health ready=%s detail=%s", ready, detail)
    return HealthResponse(
        status="ok",
        app=s.app_name,
        version=s.app_version,
        ready=ready,
        detail=detail,
    )

@app.get("/documents", response_model=DocumentsResponse)
async def documents(collection: Any = Depends(get_collection)) -> DocumentsResponse:
    docs = list_indexed_documents(collection)
    return DocumentsResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )

@app.get("/documents/{source:path}", response_model=DocumentDetailResponse)
async def document_detail(
    source: str,
    collection: Any = Depends(get_collection),
) -> DocumentDetailResponse:
    decoded = unquote(source)
    detail = get_document_detail(collection, decoded)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {decoded}")
    return DocumentDetailResponse(**detail)


@app.delete("/documents/{source:path}")
async def remove_document(source: str, collection: Any = Depends(get_collection)) -> dict[str, Any]:
    decoded = unquote(source)
    deleted = delete_document(collection, decoded)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {decoded}")
    log.info("DELETE /documents/%s removed_chunks=%s", decoded, deleted)
    return {"source": decoded, "deleted_chunks": deleted}

@app.get("/settings/chunking", response_model=ChunkingSettings)
async def get_chunking() -> ChunkingSettings:
    data = get_chunking_settings()
    return ChunkingSettings(**data)

@app.put("/settings/chunking", response_model=ChunkingSettings)
async def put_chunking(body: ChunkingSettings) -> ChunkingSettings:
    try:
        data = update_chunking_settings(body.chunk_size, body.chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ChunkingSettings(**data)

@app.get("/settings/keys", response_model=KeyConfigListResponse)
async def get_keys() -> KeyConfigListResponse:
    return KeyConfigListResponse(items=[KeyConfigItem(**item) for item in list_key_configs()])

@app.put("/settings/keys", response_model=KeyConfigItem)
async def put_key(body: KeyConfigUpsertRequest) -> KeyConfigItem:
    try:
        item = upsert_key_config(body.name, body.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return KeyConfigItem(**item)

@app.delete("/settings/keys/{name}")
async def remove_key(name: str) -> dict[str, str]:
    try:
        delete_key_config(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"name": name, "status": "deleted"}

@app.get("/settings/models", response_model=ModelsResponse)
async def get_models(s: Settings = Depends(get_settings)) -> ModelsResponse:
    models = list_available_models()
    return ModelsResponse(models=models, default_model=s.chat_model)

@app.get("/collections", response_model=CollectionsResponse)
async def collections(
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
) -> CollectionsResponse:
    docs = list_indexed_documents(collection)
    return CollectionsResponse(
        collections=[
            CollectionInfo(name=s.collection_name, document_count=len(docs)),
        ]
    )

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
            log.warning(
                "Upload rejected oversized name=%s bytes=%s max=%s",
                uf.filename,
                len(data),
                MAX_UPLOAD_BYTES,
            )
            raise HTTPException(
                status_code=413,
                detail=f"File {uf.filename} exceeds {MAX_UPLOAD_BYTES} bytes",
            )
        safe = Path(uf.filename).name
        dest = raw_dir / safe
        dest.write_bytes(data)
        paths.append(dest)
    return paths

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: Annotated[
        UploadFile,
        File(description="单个文档：.txt / .md / .pdf"),
    ],
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> IngestResponse:
    require_embedding_config(s)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    try:
        paths = await _save_uploads_to_temp([file], raw_dir)
        if not paths:
            log.warning("POST /ingest rejected: no valid file")
            raise HTTPException(status_code=400, detail="No valid file uploaded")

        log.info("POST /ingest file=%s", paths[0].name if paths else "")
        n, sources, elapsed = await ingest_documents(paths, collection, client, s)
        log.info(
            "POST /ingest done chunks=%s sources=%s elapsed_s=%s",
            n,
            sources,
            round(elapsed, 3),
        )
        return IngestResponse(indexed_chunks=n, sources=sources, seconds=round(elapsed, 3))
    finally:
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

@app.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(
    files: Annotated[
        list[UploadFile],
        File(description="多文件上传"),
    ],
    s: Settings = Depends(get_settings),
    collection: Any = Depends(get_collection),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> IngestResponse:
    require_embedding_config(s)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    try:
        paths = await _save_uploads_to_temp(files, raw_dir)
        if not paths:
            log.warning("POST /ingest/batch rejected: no valid files")
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        log.info(
            "POST /ingest/batch files=%s count=%s",
            [p.name for p in paths],
            len(paths),
        )
        n, sources, elapsed = await ingest_documents(paths, collection, client, s)
        log.info(
            "POST /ingest/batch done chunks=%s sources=%s elapsed_s=%s",
            n,
            sources,
            round(elapsed, 3),
        )
        return IngestResponse(indexed_chunks=n, sources=sources, seconds=round(elapsed, 3))
    finally:
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

@app.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    s: Settings = Depends(get_settings),
    client: OpenAICompatibleClient = Depends(get_llm_client),
) -> QueryResponse:
    require_query_config(s)

    collection_name = body.collection or s.collection_name
    collection = get_collection_by_name(collection_name)

    if body.sources is not None and len(body.sources) == 0:
        log.warning("POST /query rejected: no sources selected")
        return QueryResponse(
            answer="请至少选择一个知识库文档后再提问。",
            citations=[],
            no_relevant_context=True,
        )

    log.info(
        "POST /query question_len=%s model=%s collection=%s sources=%s",
        len(body.question),
        body.model,
        collection_name,
        len(body.sources) if body.sources else "all",
    )
    chunks = await search_relevant_chunks(
        body.question,
        collection,
        client,
        s,
        sources=body.sources,
    )

    if not chunks:
        log.warning("POST /query: no chunks retrieved")
        return QueryResponse(
            answer="知识库中暂无相关内容，请先上传并索引文档。",
            citations=[],
            no_relevant_context=True,
        )

    best = min(chunks, key=lambda c: c.distance)
    if s.max_retrieval_distance is not None and best.distance > s.max_retrieval_distance:
        log.warning(
            "POST /query: best distance exceeds threshold distance=%s max=%s",
            best.distance,
            s.max_retrieval_distance,
        )
        return QueryResponse(
            answer="在知识库中未找到与问题足够相关的可靠片段。",
            citations=[],
            no_relevant_context=True,
        )

    messages, mapping = build_messages(body.question, chunks, s)
    answer = await client.chat_completion(messages, model=body.model)
    log.info(
        "POST /query answered chunks_used=%s best_distance=%s answer_len=%s",
        len(mapping),
        best.distance,
        len(answer),
    )

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
