from pydantic import BaseModel, Field


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
