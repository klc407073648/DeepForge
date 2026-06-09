import type {
  DocumentsResponse,
  HealthResponse,
  IngestResponse,
  KeyConfigListResponse,
  ModelsResponse,
  QueryResponse,
} from '../../src/types/api'

export const healthOk: HealthResponse = {
  status: 'ok',
  app: 'rag-service',
  version: '0.1.0',
  ready: true,
  detail: null,
}

export const healthNotReady: HealthResponse = {
  ...healthOk,
  ready: false,
  detail: '请先配置 Embedding API Key',
}

export const documentsEmpty: DocumentsResponse = {
  documents: [],
  total: 0,
}

export const documentsSample: DocumentsResponse = {
  documents: [
    {
      source: 'guide.md',
      format: 'md',
      chunk_count: 5,
      uploaded_at: '2026-01-01T00:00:00Z',
      status: 'indexed',
    },
    {
      source: 'readme.txt',
      format: 'txt',
      chunk_count: 2,
      uploaded_at: '2026-01-02T00:00:00Z',
      status: 'indexed',
    },
  ],
  total: 2,
}

export const ingestSuccess: IngestResponse = {
  indexed_chunks: 3,
  sources: ['demo.txt'],
  seconds: 0.42,
}

export const keysSample: KeyConfigListResponse = {
  items: [
    {
      name: 'openai_api_key',
      label: 'OpenAI API Key',
      masked_value: 'sk-***abc',
      purpose: '通用 API Key',
      configured: true,
      source: 'env',
    },
    {
      name: 'embedding_api_key',
      label: 'Embedding API Key',
      masked_value: 'sk-***xyz',
      purpose: 'Embedding 请求',
      configured: true,
      source: 'runtime',
    },
  ],
}

export const modelsSample: ModelsResponse = {
  models: ['gpt-4o-mini', 'deepseek-chat'],
  default_model: 'gpt-4o-mini',
}

export const querySuccess: QueryResponse = {
  answer: '这是基于知识库的回答。',
  citations: [
    {
      id: 1,
      text: '引用片段内容',
      source: 'guide.md',
      distance: 0.123,
    },
  ],
  no_relevant_context: false,
}
