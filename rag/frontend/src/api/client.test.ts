import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { server } from '../../test/utils/server'
import { API_BASE } from '../../test/utils/constants'
import { ApiError, apiFetch, apiUpload } from './client'

describe('ApiError', () => {
  it('保留 status 与 message', () => {
    const err = new ApiError(404, 'not found')
    expect(err).toBeInstanceOf(Error)
    expect(err.status).toBe(404)
    expect(err.message).toBe('not found')
  })
})

describe('apiFetch', () => {
  it('成功时解析 JSON', async () => {
    const data = await apiFetch<{ ready: boolean }>('/health')
    expect(data.ready).toBe(true)
  })

  it('204 时返回 undefined', async () => {
    server.use(
      http.delete(`${API_BASE}/documents/demo.txt`, () => new HttpResponse(null, { status: 204 })),
    )
    const data = await apiFetch<undefined>('/documents/demo.txt', { method: 'DELETE' })
    expect(data).toBeUndefined()
  })

  it('FastAPI string detail 转为 ApiError', async () => {
    server.use(
      http.get(`${API_BASE}/health`, () =>
        HttpResponse.json({ detail: 'service unavailable' }, { status: 503 }),
      ),
    )
    await expect(apiFetch('/health')).rejects.toMatchObject({
      status: 503,
      message: 'service unavailable',
    })
  })

  it('FastAPI validation detail 数组拼接为 ApiError', async () => {
    server.use(
      http.post(`${API_BASE}/query`, () =>
        HttpResponse.json(
          { detail: [{ msg: 'field required' }, { msg: 'invalid model' }] },
          { status: 422 },
        ),
      ),
    )
    await expect(apiFetch('/query', { method: 'POST' })).rejects.toMatchObject({
      status: 422,
      message: 'field required; invalid model',
    })
  })

  it('非 JSON 错误体回落到 statusText', async () => {
    server.use(
      http.get(`${API_BASE}/health`, () =>
        new HttpResponse('plain text error', { status: 500, statusText: 'Internal Server Error' }),
      ),
    )
    await expect(apiFetch('/health')).rejects.toMatchObject({
      status: 500,
      message: 'Internal Server Error',
    })
  })
})

describe('apiUpload', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('单文件 POST /ingest 且字段名为 file', async () => {
    let capturedBody: BodyInit | null | undefined
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      capturedBody = init?.body
      return Response.json({ indexed_chunks: 1, sources: ['a.txt'], seconds: 0.1 })
    })

    const file = new File(['hello'], 'a.txt', { type: 'text/plain' })
    const result = await apiUpload<{ indexed_chunks: number }>([file])

    expect(String(fetchMock.mock.calls[0]?.[0]).endsWith('/api/ingest')).toBe(true)
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBe('POST')
    expect(result.indexed_chunks).toBe(1)

    const form = capturedBody as FormData
    expect(form.get('file')).toBeInstanceOf(File)
    expect((form.get('file') as File).name).toBe('a.txt')
    expect(form.get('files')).toBeNull()
  })

  it('多文件 POST /ingest/batch 且字段名为 files', async () => {
    let capturedBody: BodyInit | null | undefined
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      capturedBody = init?.body
      return Response.json({
        indexed_chunks: 2,
        sources: ['a.txt', 'b.txt'],
        seconds: 0.2,
      })
    })

    const files = [
      new File(['a'], 'a.txt', { type: 'text/plain' }),
      new File(['b'], 'b.txt', { type: 'text/plain' }),
    ]
    await apiUpload(files)

    expect(String(fetchMock.mock.calls[0]?.[0]).endsWith('/api/ingest/batch')).toBe(true)

    const form = capturedBody as FormData
    expect(form.get('file')).toBeNull()
    expect(form.getAll('files')).toHaveLength(2)
  })
})
