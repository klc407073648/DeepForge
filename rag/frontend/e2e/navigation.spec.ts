import { expect, test } from '@playwright/test'

test.describe('页面导航', () => {
  test('默认进入知识库页', async ({ page }) => {
    await page.goto('/')

    await expect(page.getByRole('heading', { name: '知识库' })).toBeVisible()
    await expect(page.getByRole('link', { name: '知识库' })).toHaveClass(/bg-white/)
  })

  test('可切换到智能问答与 API Key 管理', async ({ page }) => {
    await page.goto('/knowledge')

    await page.getByRole('link', { name: '智能问答' }).click()
    await expect(page.getByRole('heading', { name: '智能问答' })).toBeVisible()

    await page.getByRole('link', { name: 'API Key 管理' }).click()
    await expect(page.getByRole('heading', { name: 'API Key 管理' })).toBeVisible()
  })

  test('侧边栏展示版本与健康状态', async ({ page }) => {
    await page.goto('/knowledge')

    const aside = page.locator('aside')
    await expect(aside.getByText(/v\d+\.\d+\.\d+/)).toBeVisible()
    await expect(aside.getByText(/服务就绪|服务未就绪/)).toBeVisible()
  })
})
