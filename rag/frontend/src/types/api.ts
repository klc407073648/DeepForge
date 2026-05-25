export interface HealthResponse {
  status: string
  app: string
  version: string
  ready: boolean
  detail?: string | null
}

export interface DocumentInfo {
  source: string
  format: string
  chunk_count: number
  uploaded_at: string
  status: string
}

export interface DocumentsResponse {
  documents: DocumentInfo[]
  total: number
}

export interface ChunkInfo {
  id: string
  chunk_index: number
  text: string
  source: string
}

export interface DocumentDetailResponse {
  source: string
  format: string
  chunk_count: number
  status: string
  chunks: ChunkInfo[]
}

export interface IngestResponse {
  indexed_chunks: number
  sources: string[]
  seconds: number
}

export interface ChunkingSettings {
  chunk_size: number
  chunk_overlap: number
}

export interface CollectionInfo {
  name: string
  document_count: number
}

export interface CollectionsResponse {
  collections: CollectionInfo[]
}

export interface KeyConfigItem {
  name: string
  label: string
  masked_value: string
  purpose: string
  configured: boolean
  source: string
}

export interface KeyConfigListResponse {
  items: KeyConfigItem[]
}

export interface ModelsResponse {
  models: string[]
  default_model: string
}

export interface Citation {
  id: number
  text: string
  source: string
  distance?: number | null
}

export interface QueryRequest {
  question: string
  model?: string | null
  collection?: string | null
  sources?: string[] | null
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  no_relevant_context: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}
