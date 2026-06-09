import { expect, test } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { BACKEND_NOT_READY_MSG, isBackendReady } from './helpers'

const fixturePath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'sample.txt')
const uploadedName = 'sample.txt'

test.describe.configure({ mode: 'serial' })

test.describe('知识库（需后端就绪）', () => {
  let backendReady = false

  test.beforeAll(async ({ request }) => {
    backendReady = await isBackendReady(request)
  })

  test.beforeEach(() => {
    test.skip(!backendReady, BACKEND_NOT_READY_MSG)
  })

  test('上传文档并出现在列表中', async ({ page }) => {
    await page.goto('/knowledge')

    await expect(page.getByRole('button', { name: '上传文档' })).toBeEnabled()

    await page.getByRole('button', { name: '上传文档' }).click()
    await page.locator('input[type="file"][multiple]').setInputFiles(fixturePath)
    await page.getByRole('button', { name: '开始索引' }).click()

    await expect(page.getByText(/索引完成：\d+ 个片段/)).toBeVisible({ timeout: 120_000 })
    await expect(page.getByRole('cell', { name: uploadedName })).toBeVisible()
  })

  test('删除已上传文档', async ({ page }) => {
    await page.goto('/knowledge')

    const row = page.getByRole('row').filter({ hasText: uploadedName })
    await expect(row).toBeVisible({ timeout: 30_000 })

    page.once('dialog', (dialog) => dialog.accept())
    await row.getByTitle('删除').click()

    await expect(page.getByText('文档已删除')).toBeVisible()
    await expect(page.getByRole('cell', { name: uploadedName })).toHaveCount(0)
  })
})
