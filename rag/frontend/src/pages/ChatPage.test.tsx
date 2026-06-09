import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'
import userEvent from '@testing-library/user-event'
import { screen, waitFor } from '@testing-library/react'
import ChatPage from './ChatPage'
import {
  documentsSample,
  healthNotReady,
  querySuccess,
} from '../../test/fixtures/api'
import { API_BASE } from '../../test/utils/constants'
import { renderWithProviders } from '../../test/utils/render'
import { server } from '../../test/utils/server'

describe('ChatPage', () => {
  it('无文档时禁用知识库选择与输入', async () => {
    renderWithProviders(<ChatPage />)

    expect(await screen.findByText('暂无文档')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('输入你的问题，Shift+Enter 换行')).toBeDisabled()
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
  })

  it('服务未就绪时禁用输入并展示状态', async () => {
    server.use(
      http.get(`${API_BASE}/health`, () => HttpResponse.json(healthNotReady)),
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
    )

    renderWithProviders(<ChatPage />)

    expect(await screen.findByText('服务未就绪')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('输入你的问题，Shift+Enter 换行')).toBeDisabled()
  })

  it('清空文档选择后禁用输入与发送', async () => {
    server.use(http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)))

    const user = userEvent.setup()
    renderWithProviders(<ChatPage />)

    expect(await screen.findByText('全部文档（2 个）')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /全部文档/ }))
    await user.click(screen.getByRole('button', { name: '清空' }))

    expect(screen.getByText('已选 0/2 个文档')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('输入你的问题，Shift+Enter 换行')).toBeDisabled()
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
  })

  it('问答成功时展示回答与引用来源', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
      http.post(`${API_BASE}/query`, () => HttpResponse.json(querySuccess)),
    )

    const user = userEvent.setup()
    renderWithProviders(<ChatPage />)

    await screen.findByText('全部文档（2 个）')

    await user.type(screen.getByPlaceholderText('输入你的问题，Shift+Enter 换行'), '什么是 RAG？')
    await user.click(screen.getByRole('button', { name: '发送' }))

    expect(await screen.findByText(querySuccess.answer)).toBeInTheDocument()
    expect(screen.getByText('引用来源 (1)')).toBeInTheDocument()
    expect(screen.getByText('引用片段内容')).toBeInTheDocument()
    expect(screen.getByText('guide.md')).toBeInTheDocument()
  })

  it('问答失败时展示 ApiError 信息', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
      http.post(`${API_BASE}/query`, () =>
        HttpResponse.json({ detail: 'chat model unavailable' }, { status: 503 }),
      ),
    )

    const user = userEvent.setup()
    renderWithProviders(<ChatPage />)

    await screen.findByText('全部文档（2 个）')

    await user.type(screen.getByPlaceholderText('输入你的问题，Shift+Enter 换行'), '测试问题')
    await user.click(screen.getByRole('button', { name: '发送' }))

    expect(await screen.findByText('chat model unavailable')).toBeInTheDocument()
  })

  it('默认选中全部文档并加载模型列表', async () => {
    server.use(http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)))

    renderWithProviders(<ChatPage />)

    expect(await screen.findByText('全部文档（2 个）')).toBeInTheDocument()

    const modelSelect = screen.getByRole('combobox')
    await waitFor(() => {
      expect(modelSelect).toHaveValue('gpt-4o-mini')
    })
    expect(screen.getByRole('option', { name: 'deepseek-chat' })).toBeInTheDocument()
  })
})
