import { expect, test } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { BACKEND_NOT_READY_MSG, isBackendReady } from './helpers'

const fixturePath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'sample.txt')

test.describe.configure({ mode: 'serial' })

test.describe('RAG 完整流程（需后端就绪）', () => {
  let backendReady = false

  test.beforeAll(async ({ request }) => {
    backendReady = await isBackendReady(request)
  })

  test.beforeEach(() => {
    test.skip(!backendReady, BACKEND_NOT_READY_MSG)
  })

  test('上传文档建索引', async ({ page }) => {
    await page.goto('/knowledge')

    await page.getByRole('button', { name: '上传文档' }).click()
    await page.locator('input[type="file"][multiple]').setInputFiles(fixturePath)
    await page.getByRole('button', { name: '开始索引' }).click()

    await expect(page.getByText(/索引完成：\d+ 个片段/)).toBeVisible({ timeout: 120_000 })
    await expect(page.getByRole('cell', { name: 'sample.txt' })).toBeVisible()
  })

  test('智能问答返回回答与引用', async ({ page }) => {
    await page.goto('/chat')

    await expect(page.getByText(/全部文档|已选 \d+\/\d+ 个文档/)).toBeVisible()

    const textarea = page.getByPlaceholder('输入你的问题，Shift+Enter 换行')
    await expect(textarea).toBeEnabled()

    await textarea.fill('Playwright E2E 向量索引是什么？')
    await page.getByRole('button', { name: '发送' }).click()

    await expect(page.getByText('正在生成回答...')).toBeVisible()
    await expect(page.getByText('正在生成回答...')).toBeHidden({ timeout: 120_000 })

    await expect(page.getByText('Playwright E2E 向量索引是什么？')).toBeVisible()
    await expect(page.getByText(/引用来源 \(\d+\)/)).toBeVisible()
    await expect(page.getByText('sample.txt').first()).toBeVisible()
  })
})
