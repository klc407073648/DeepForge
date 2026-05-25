from pydantic import BaseModel, Field

# 请求/响应 Pydantic 模型 ↔ OpenAPI /docs
# Pydantic 模型定义了 API 与客户端之间的“契约”：
# 请求模型 (Request Model)：定义了客户端允许发送的数据结构和规则。
# 响应模型 (Response Model)：定义了客户端将会收到的数据结构和内容。
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    ready: bool
    detail: str | None = None


class IngestResponse(BaseModel):
    indexed_chunks: int
    sources: list[str]
    seconds: float


class Citation(BaseModel):
    id: int
    text: str
    source: str
    distance: float | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    model: str | None = Field(default=None, max_length=128)
    collection: str | None = Field(default=None, max_length=128)
    sources: list[str] | None = Field(default=None, max_length=256)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    no_relevant_context: bool = False


class DocumentInfo(BaseModel):
    source: str
    format: str
    chunk_count: int
    uploaded_at: str
    status: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class ChunkInfo(BaseModel):
    id: str
    chunk_index: int
    text: str
    source: str


class DocumentDetailResponse(BaseModel):
    source: str
    format: str
    chunk_count: int
    status: str
    chunks: list[ChunkInfo]


class ChunkingSettings(BaseModel):
    chunk_size: int = Field(..., ge=64, le=8192)
    chunk_overlap: int = Field(..., ge=0)


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class KeyConfigItem(BaseModel):
    name: str
    label: str
    masked_value: str
    purpose: str
    configured: bool
    source: str


class KeyConfigListResponse(BaseModel):
    items: list[KeyConfigItem]


class KeyConfigUpsertRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    value: str = Field(default="")


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: str
