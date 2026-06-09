import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import { screen, waitFor, within } from '@testing-library/react'
import ApiKeyPage from './ApiKeyPage'
import { keysSample } from '../../test/fixtures/api'
import { API_BASE } from '../../test/utils/constants'
import { renderWithProviders } from '../../test/utils/render'
import { server } from '../../test/utils/server'

describe('ApiKeyPage', () => {
  it('渲染 Key 列表与脱敏值', async () => {
    server.use(http.get(`${API_BASE}/settings/keys`, () => HttpResponse.json(keysSample)))

    renderWithProviders(<ApiKeyPage />)

    expect(await screen.findByText('OpenAI API Key')).toBeInTheDocument()
    expect(screen.getByText('sk-***abc')).toBeInTheDocument()
    expect(screen.getByText('sk-***xyz')).toBeInTheDocument()
    expect(screen.getAllByText('已配置')).toHaveLength(2)
    expect(screen.getByText('运行时')).toBeInTheDocument()
  })

  it('保存 Key 后展示成功提示', async () => {
    server.use(
      http.get(`${API_BASE}/settings/keys`, () => HttpResponse.json(keysSample)),
      http.put(`${API_BASE}/settings/keys`, async ({ request }) => {
        const body = (await request.json()) as { name: string; value: string }
        return HttpResponse.json({
          name: body.name,
          label: 'OpenAI API Key',
          masked_value: 'sk-***new',
          purpose: '通用 API Key',
          configured: true,
          source: 'runtime',
        })
      }),
    )

    const user = userEvent.setup()
    renderWithProviders(<ApiKeyPage />)

    const row = (await screen.findByText('OpenAI API Key')).closest('tr')!
    await user.click(within(row).getAllByRole('button')[0]!)

    expect(await screen.findByText('编辑 OpenAI API Key')).toBeInTheDocument()

    await user.type(screen.getByPlaceholderText('输入新值'), 'sk-test-new-key')
    await user.click(screen.getByRole('button', { name: '保存' }))

    expect(await screen.findByText('配置已保存')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.queryByText('编辑 OpenAI API Key')).not.toBeInTheDocument()
    })
  })

  it('清除运行时覆盖并提示回退到 .env', async () => {
    server.use(
      http.get(`${API_BASE}/settings/keys`, () => HttpResponse.json(keysSample)),
      http.delete(`${API_BASE}/settings/keys/embedding_api_key`, () =>
        HttpResponse.json({ name: 'embedding_api_key', status: 'deleted' }),
      ),
    )

    vi.spyOn(window, 'confirm').mockReturnValue(true)

    const user = userEvent.setup()
    renderWithProviders(<ApiKeyPage />)

    const row = (await screen.findByText('Embedding API Key')).closest('tr')!
    const buttons = within(row).getAllByRole('button')
    await user.click(buttons[buttons.length - 1]!)

    expect(await screen.findByText('运行时覆盖已清除，将回退到 .env 配置')).toBeInTheDocument()
  })

  it('保存失败时展示 ApiError 信息', async () => {
    server.use(
      http.get(`${API_BASE}/settings/keys`, () => HttpResponse.json(keysSample)),
      http.put(`${API_BASE}/settings/keys`, () =>
        HttpResponse.json({ detail: 'invalid key format' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    renderWithProviders(<ApiKeyPage />)

    const row = (await screen.findByText('OpenAI API Key')).closest('tr')!
    await user.click(within(row).getAllByRole('button')[0]!)
    await user.type(screen.getByPlaceholderText('输入新值'), 'bad')
    await user.click(screen.getByRole('button', { name: '保存' }))

    expect(await screen.findByText('invalid key format')).toBeInTheDocument()
  })
})
