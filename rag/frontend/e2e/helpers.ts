import type { APIRequestContext, Page } from '@playwright/test'
import { expect } from '@playwright/test'

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

/** 等待页面 health 请求完成且上传按钮可用 */
export async function waitForKnowledgeReady(page: Page) {
  await expect(page.getByRole('button', { name: '上传文档' })).toBeEnabled({ timeout: 30_000 })
}

export const BACKEND_NOT_READY_MSG =
  '后端未就绪：请在 rag/.env 配置 API Key，或设置 EMBEDDING_BACKEND=local 后重试'
