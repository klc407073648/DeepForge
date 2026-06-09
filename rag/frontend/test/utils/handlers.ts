import { http, HttpResponse } from 'msw'
import { documentsEmpty, healthOk, ingestSuccess, modelsSample } from '../fixtures/api'
import { API_BASE } from './constants'

/**
 * 默认 MSW handlers，覆盖常见只读端点。
 * 组件测试与 mutation 用例请用 server.use(...) 按用例覆盖。
 */
export const handlers = [
  http.get(`${API_BASE}/health`, () => HttpResponse.json(healthOk)),

  http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsEmpty)),

  http.get(`${API_BASE}/settings/chunking`, () =>
    HttpResponse.json({ chunk_size: 512, chunk_overlap: 128 }),
  ),

  http.get(`${API_BASE}/settings/keys`, () => HttpResponse.json({ items: [] })),

  http.get(`${API_BASE}/settings/models`, () => HttpResponse.json(modelsSample)),

  http.get(`${API_BASE}/collections`, () => HttpResponse.json({ collections: [] })),

  http.post(`${API_BASE}/ingest`, () => HttpResponse.json(ingestSuccess)),

  http.post(`${API_BASE}/ingest/batch`, () => HttpResponse.json(ingestSuccess)),
]
