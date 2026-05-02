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


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    no_relevant_context: bool = False
