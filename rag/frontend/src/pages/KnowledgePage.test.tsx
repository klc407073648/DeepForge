import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import { screen, within } from '@testing-library/react'
import KnowledgePage from './KnowledgePage'
import { documentsSample, healthNotReady, ingestSuccess } from '../../test/fixtures/api'
import { API_BASE } from '../../test/utils/constants'
import { renderWithProviders } from '../../test/utils/render'
import { server } from '../../test/utils/server'

describe('KnowledgePage', () => {
  it('渲染文档列表', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
    )

    renderWithProviders(<KnowledgePage />)

    expect(await screen.findByText('guide.md')).toBeInTheDocument()
    expect(screen.getByText('readme.txt')).toBeInTheDocument()
  })

  it('按名称搜索过滤文档', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
    )

    const user = userEvent.setup()
    renderWithProviders(<KnowledgePage />)

    await screen.findByText('guide.md')

    await user.type(screen.getByPlaceholderText('搜索文档名称'), 'guide')

    expect(screen.getByText('guide.md')).toBeInTheDocument()
    expect(screen.queryByText('readme.txt')).not.toBeInTheDocument()
  })

  it('服务未就绪时展示提示并禁用上传', async () => {
    server.use(
      http.get(`${API_BASE}/health`, () => HttpResponse.json(healthNotReady)),
    )

    renderWithProviders(<KnowledgePage />)

    expect(await screen.findByText(/服务未就绪/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /上传文档/ })).toBeDisabled()
  })

  it('超大文件被忽略并提示错误', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
    )

    const user = userEvent.setup()
    renderWithProviders(<KnowledgePage />)

    await screen.findByText('guide.md')
    await user.click(screen.getByRole('button', { name: /上传文档/ }))

    const bigFile = new File(['x'], 'big.pdf', { type: 'application/pdf' })
    Object.defineProperty(bigFile, 'size', { value: 15 * 1024 * 1024 + 1 })

    const fileInput = document.querySelector('input[type="file"][multiple]') as HTMLInputElement
    await user.upload(fileInput, bigFile)

    expect(await screen.findByText('部分文件超过 15MB 已被忽略')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始索引' })).toBeDisabled()
  })

  it('上传成功后展示索引结果', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
      http.post(`${API_BASE}/ingest`, () => HttpResponse.json(ingestSuccess)),
    )

    const user = userEvent.setup()
    renderWithProviders(<KnowledgePage />)

    await screen.findByText('guide.md')
    await user.click(screen.getByRole('button', { name: /上传文档/ }))

    const file = new File(['hello world'], 'demo.txt', { type: 'text/plain' })
    const fileInput = document.querySelector('input[type="file"][multiple]') as HTMLInputElement
    await user.upload(fileInput, file)

    expect(screen.getByText('demo.txt')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '开始索引' }))

    expect(
      await screen.findByText(`索引完成：${ingestSuccess.indexed_chunks} 个片段，耗时 ${ingestSuccess.seconds}s`),
    ).toBeInTheDocument()
  })

  it('确认后删除文档并提示成功', async () => {
    server.use(
      http.get(`${API_BASE}/documents`, () => HttpResponse.json(documentsSample)),
      http.delete(`${API_BASE}/documents/guide.md`, () => new HttpResponse(null, { status: 204 })),
    )

    vi.spyOn(window, 'confirm').mockReturnValue(true)

    const user = userEvent.setup()
    renderWithProviders(<KnowledgePage />)

    const row = (await screen.findByText('guide.md')).closest('tr')!
    await user.click(within(row).getByTitle('删除'))

    expect(await screen.findByText('文档已删除')).toBeInTheDocument()
  })
})
