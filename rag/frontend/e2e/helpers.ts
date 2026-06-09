import type { APIRequestContext } from '@playwright/test'

const BACKEND_HEALTH_URL = 'http://127.0.0.1:8000/health'

export async function isBackendReady(request: APIRequestContext): Promise<boolean> {
  try {
    const res = await request.get(BACKEND_HEALTH_URL)
    if (!res.ok()) return false
    const data = (await res.json()) as { ready?: boolean }
    return data.ready === true
  } catch {
    return false
  }
}

export const BACKEND_NOT_READY_MSG =
  '后端未就绪：请在 rag/.env 配置 API Key，或设置 EMBEDDING_BACKEND=local 后重试'
