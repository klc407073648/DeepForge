const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail.map((d: { msg?: string }) => d.msg ?? '').join('; ')
    }
    return res.statusText
  } catch {
    return res.statusText
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    throw new ApiError(res.status, await parseError(res))
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export async function apiUpload<T>(files: File[]): Promise<T> {
  const form = new FormData()
  if (files.length === 1) {
    form.append('file', files[0])
  } else {
    files.forEach((f) => form.append('files', f))
  }
  const endpoint = files.length === 1 ? '/ingest' : '/ingest/batch'
  return apiFetch<T>(endpoint, { method: 'POST', body: form })
}
