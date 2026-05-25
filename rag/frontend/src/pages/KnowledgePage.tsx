import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Pencil, Eye, Plus, Search, Trash2, Upload, X } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import {
  deleteDocument,
  getChunking,
  getDocumentDetail,
  getDocuments,
  getHealth,
  ingestFiles,
  updateChunking,
} from '../api/rag'
import { ApiError } from '../api/client'

const ACCEPT = '.txt,.md,.markdown,.pdf'
const MAX_BYTES = 15 * 1024 * 1024

export default function KnowledgePage() {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const replaceInputRef = useRef<HTMLInputElement>(null)
  const [search, setSearch] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [showUpload, setShowUpload] = useState(false)
  const [replaceSource, setReplaceSource] = useState<string | null>(null)
  const [detailSource, setDetailSource] = useState<string | null>(null)
  const [chunkSize, setChunkSize] = useState(512)
  const [chunkOverlap, setChunkOverlap] = useState(128)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: getHealth })
  const { data: docsData, isLoading } = useQuery({ queryKey: ['documents'], queryFn: getDocuments })

  const {
    data: detailData,
    isLoading: detailLoading,
    error: detailError,
  } = useQuery({
    queryKey: ['document-detail', detailSource],
    queryFn: () => getDocumentDetail(detailSource!),
    enabled: !!detailSource,
  })

  useQuery({
    queryKey: ['chunking'],
    queryFn: async () => {
      const data = await getChunking()
      setChunkSize(data.chunk_size)
      setChunkOverlap(data.chunk_overlap)
      return data
    },
  })

  const filteredDocs = useMemo(() => {
    const docs = docsData?.documents ?? []
    if (!search.trim()) return docs
    const q = search.toLowerCase()
    return docs.filter((d) => d.source.toLowerCase().includes(q))
  }, [docsData, search])

  const uploadMutation = useMutation({
    mutationFn: ingestFiles,
    onSuccess: (res) => {
      setMessage(`索引完成：${res.indexed_chunks} 个片段，耗时 ${res.seconds}s`)
      setSelectedFiles([])
      setShowUpload(false)
      setReplaceSource(null)
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['collections'] })
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : '上传失败'),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      setMessage('文档已删除')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['collections'] })
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : '删除失败'),
  })

  const chunkingMutation = useMutation({
    mutationFn: updateChunking,
    onSuccess: () => setMessage('切片设置已保存（仅对新上传文档生效）'),
    onError: (e) => setError(e instanceof ApiError ? e.message : '保存失败'),
  })

  const handleFiles = (files: FileList | null) => {
    if (!files) return
    const valid = Array.from(files).filter((f) => f.size <= MAX_BYTES)
    setSelectedFiles(valid)
    setError(valid.length < files.length ? '部分文件超过 15MB 已被忽略' : null)
  }

  const handleUpload = () => {
    if (selectedFiles.length === 0) return
    uploadMutation.mutate(selectedFiles)
  }

  const handleReplace = (source: string) => {
    setReplaceSource(source)
    replaceInputRef.current?.click()
  }

  const handleReplaceFile = (files: FileList | null) => {
    if (!files?.[0]) return
    uploadMutation.mutate([files[0]])
  }

  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900">知识库</h1>
        <p className="mt-2 text-sm text-gray-500">管理已上传文档，配置切片参数并建立向量索引。</p>
      </div>

      {!health?.ready && (
        <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          服务未就绪：{health?.detail ?? '请先配置 Embedding API Key'}
        </div>
      )}

      {message && (
        <div className="mb-4 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-800">
          {message}
        </div>
      )}
      {error && (
        <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <div className="mb-6 flex items-center justify-between gap-4">
        <div className="relative max-w-sm flex-1">
          <Search className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={16} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索文档名称"
            className="w-full rounded-xl border border-gray-200 bg-white py-2 pr-3 pl-9 text-sm outline-none focus:border-gray-400"
          />
        </div>
        <button
          type="button"
          disabled={!health?.ready}
          onClick={() => setShowUpload(true)}
          className="inline-flex items-center gap-2 rounded-xl bg-gray-900 px-4 py-2 text-sm text-white disabled:opacity-50"
        >
          <Plus size={16} />
          上传文档
        </button>
      </div>

      <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-100 bg-gray-50 text-gray-500">
            <tr>
              <th className="px-4 py-3 font-medium">文档名称</th>
              <th className="px-4 py-3 font-medium">格式</th>
              <th className="px-4 py-3 font-medium">片段数</th>
              <th className="px-4 py-3 font-medium">状态</th>
              <th className="px-4 py-3 font-medium">操作</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-400">
                  加载中...
                </td>
              </tr>
            )}
            {!isLoading && filteredDocs.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-gray-400">
                  暂无文档，点击「上传文档」开始建立知识库
                </td>
              </tr>
            )}
            {filteredDocs.map((doc) => (
              <tr key={doc.source} className="border-b border-gray-50 hover:bg-gray-50/50">
                <td className="px-4 py-3 font-medium text-gray-900">{doc.source}</td>
                <td className="px-4 py-3 text-gray-600">{doc.format}</td>
                <td className="px-4 py-3 text-gray-600">{doc.chunk_count}</td>
                <td className="px-4 py-3">
                  <span className="rounded-full bg-green-50 px-2 py-0.5 text-xs text-green-700">
                    {doc.status}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      title="查看详情"
                      onClick={() => setDetailSource(doc.source)}
                      className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100"
                    >
                      <Eye size={16} />
                    </button>
                    <button
                      type="button"
                      title="重新上传覆盖"
                      onClick={() => handleReplace(doc.source)}
                      className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100"
                    >
                      <Pencil size={16} />
                    </button>
                    <button
                      type="button"
                      title="删除"
                      onClick={() => {
                        if (window.confirm(`确认删除 ${doc.source}？`)) {
                          deleteMutation.mutate(doc.source)
                        }
                      }}
                      className="rounded-lg p-1.5 text-gray-500 hover:bg-red-50 hover:text-red-600"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-8 rounded-2xl border border-gray-200 bg-white p-6">
        <h2 className="mb-4 text-lg font-medium text-gray-900">切片设置</h2>
        <div className="grid max-w-xl grid-cols-2 gap-4">
          <label className="text-sm text-gray-600">
            Chunk Size
            <input
              type="number"
              min={64}
              max={8192}
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 outline-none focus:border-gray-400"
            />
          </label>
          <label className="text-sm text-gray-600">
            Chunk Overlap
            <input
              type="number"
              min={0}
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Number(e.target.value))}
              className="mt-1 w-full rounded-xl border border-gray-200 px-3 py-2 outline-none focus:border-gray-400"
            />
          </label>
        </div>
        <button
          type="button"
          onClick={() => chunkingMutation.mutate({ chunk_size: chunkSize, chunk_overlap: chunkOverlap })}
          className="mt-4 rounded-xl bg-gray-900 px-4 py-2 text-sm text-white"
        >
          保存设置
        </button>
      </div>

      {detailSource && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
          <div className="flex max-h-[85vh] w-full max-w-3xl flex-col rounded-2xl bg-white shadow-xl">
            <div className="flex items-start justify-between border-b border-gray-100 px-6 py-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900">文档详情</h3>
                <p className="mt-1 text-sm text-gray-500">{detailSource}</p>
              </div>
              <button
                type="button"
                onClick={() => setDetailSource(null)}
                className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              >
                <X size={18} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-6 py-4">
              {detailLoading && (
                <p className="py-8 text-center text-sm text-gray-400">加载中...</p>
              )}
              {detailError && (
                <p className="py-8 text-center text-sm text-red-600">
                  {detailError instanceof ApiError ? detailError.message : '加载失败'}
                </p>
              )}
              {detailData && (
                <>
                  <div className="mb-6 grid grid-cols-3 gap-4 rounded-xl border border-gray-100 bg-gray-50 p-4 text-sm">
                    <div>
                      <p className="text-gray-500">格式</p>
                      <p className="mt-1 font-medium text-gray-900">{detailData.format}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">片段数</p>
                      <p className="mt-1 font-medium text-gray-900">{detailData.chunk_count}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">状态</p>
                      <p className="mt-1">
                        <span className="rounded-full bg-green-50 px-2 py-0.5 text-xs text-green-700">
                          {detailData.status}
                        </span>
                      </p>
                    </div>
                  </div>

                  <h4 className="mb-3 text-sm font-medium text-gray-900">
                    分片内容（共 {detailData.chunks.length} 个）
                  </h4>
                  <div className="space-y-3">
                    {detailData.chunks.map((chunk) => (
                      <div
                        key={chunk.id}
                        className="rounded-xl border border-gray-200 bg-white p-4"
                      >
                        <div className="mb-2 flex items-center justify-between">
                          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
                            片段 #{chunk.chunk_index + 1}
                          </span>
                          <span className="text-xs text-gray-400">{chunk.text.length} 字符</span>
                        </div>
                        <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-gray-700">
                          {chunk.text}
                        </pre>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="border-t border-gray-100 px-6 py-4">
              <button
                type="button"
                onClick={() => setDetailSource(null)}
                className="rounded-xl bg-gray-900 px-4 py-2 text-sm text-white"
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {showUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
          <div className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl">
            <h3 className="mb-4 text-lg font-medium">上传文档</h3>
            <div
              className="flex cursor-pointer flex-col items-center rounded-2xl border-2 border-dashed border-gray-200 px-6 py-10 text-center"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="mb-3 text-gray-400" size={32} />
              <p className="text-sm text-gray-700">拖拽或点击选择文件</p>
              <p className="mt-1 text-xs text-gray-400">支持 .txt / .md / .pdf，单文件最大 15MB</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPT}
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            {selectedFiles.length > 0 && (
              <ul className="mt-4 space-y-1 text-sm text-gray-600">
                {selectedFiles.map((f) => (
                  <li key={f.name}>{f.name}</li>
                ))}
              </ul>
            )}
            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => {
                  setShowUpload(false)
                  setSelectedFiles([])
                }}
                className="rounded-xl px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
              >
                取消
              </button>
              <button
                type="button"
                disabled={selectedFiles.length === 0 || uploadMutation.isPending}
                onClick={handleUpload}
                className="rounded-xl bg-gray-900 px-4 py-2 text-sm text-white disabled:opacity-50"
              >
                {uploadMutation.isPending ? '索引中...' : '开始索引'}
              </button>
            </div>
          </div>
        </div>
      )}

      <input
        ref={replaceInputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => handleReplaceFile(e.target.files)}
      />
      {replaceSource && (
        <span className="hidden">替换: {replaceSource}</span>
      )}
    </div>
  )
}
