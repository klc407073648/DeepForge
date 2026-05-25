import { apiFetch, apiUpload } from './client'
import type {
  ChunkingSettings,
  CollectionsResponse,
  DocumentDetailResponse,
  DocumentsResponse,
  HealthResponse,
  IngestResponse,
  KeyConfigItem,
  KeyConfigListResponse,
  ModelsResponse,
  QueryRequest,
  QueryResponse,
} from '../types/api'

export const getHealth = () => apiFetch<HealthResponse>('/health')

export const getDocuments = () => apiFetch<DocumentsResponse>('/documents')

export const getDocumentDetail = (source: string) =>
  apiFetch<DocumentDetailResponse>(`/documents/${encodeURIComponent(source)}`)

export const deleteDocument = (source: string) =>
  apiFetch<{ source: string; deleted_chunks: number }>(
    `/documents/${encodeURIComponent(source)}`,
    { method: 'DELETE' },
  )

export const ingestFiles = (files: File[]) => apiUpload<IngestResponse>(files)

export const getChunking = () => apiFetch<ChunkingSettings>('/settings/chunking')

export const updateChunking = (body: ChunkingSettings) =>
  apiFetch<ChunkingSettings>('/settings/chunking', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const getKeys = () => apiFetch<KeyConfigListResponse>('/settings/keys')

export const upsertKey = (name: string, value: string) =>
  apiFetch<KeyConfigItem>('/settings/keys', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, value }),
  })

export const deleteKey = (name: string) =>
  apiFetch<{ name: string; status: string }>(`/settings/keys/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })

export const getModels = () => apiFetch<ModelsResponse>('/settings/models')

export const getCollections = () => apiFetch<CollectionsResponse>('/collections')

export const queryRag = (body: QueryRequest) =>
  apiFetch<QueryResponse>('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
