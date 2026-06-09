import { useMutation, useQuery } from '@tanstack/react-query'
import { ChevronDown, Send } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { getDocuments, getHealth, getModels, queryRag } from '../api/rag'
import { ApiError } from '../api/client'
import type { ChatMessage, Citation, DocumentInfo } from '../types/api'

const EMPTY_DOCUMENTS: DocumentInfo[] = []

export default function ChatPage() {
  const [question, setQuestion] = useState('')
  const [model, setModel] = useState('')
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [kbOpen, setKbOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeCitation, setActiveCitation] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const kbRef = useRef<HTMLDivElement>(null)

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: getHealth })
  const { data: modelsData } = useQuery({ queryKey: ['models'], queryFn: getModels })
  const { data: docsData } = useQuery({ queryKey: ['documents'], queryFn: getDocuments })

  const models = modelsData?.models ?? []
  const documents = docsData?.documents ?? EMPTY_DOCUMENTS
  const selectedModel = model || modelsData?.default_model || models[0] || ''

  useEffect(() => {
    const all = documents.map((d) => d.source)
    setSelectedSources((prev) => {
      if (prev.length === 0) {
        if (all.length === 0) return prev
        return all
      }
      const next = prev.filter((s) => all.includes(s))
      if (next.length > 0) {
        if (next.length === prev.length && next.every((s, i) => s === prev[i])) return prev
        return next
      }
      if (all.length === 0) return prev
      if (all.length === prev.length && all.every((s, i) => s === prev[i])) return prev
      return all
    })
  }, [documents])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (kbRef.current && !kbRef.current.contains(e.target as Node)) {
        setKbOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const kbLabel = useMemo(() => {
    if (documents.length === 0) return '暂无文档'
    if (selectedSources.length === documents.length) {
      return `全部文档（${documents.length} 个）`
    }
    return `已选 ${selectedSources.length}/${documents.length} 个文档`
  }, [documents.length, selectedSources.length])

  const toggleSource = (source: string) => {
    setSelectedSources((prev) =>
      prev.includes(source) ? prev.filter((s) => s !== source) : [...prev, source],
    )
  }

  const selectAllSources = () => setSelectedSources(documents.map((d) => d.source))
  const clearAllSources = () => setSelectedSources([])

  const latestCitations = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && messages[i].citations?.length) {
        return messages[i].citations!
      }
    }
    return [] as Citation[]
  }, [messages])

  const queryMutation = useMutation({
    mutationFn: queryRag,
    onSuccess: (res, vars) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.answer, citations: res.citations },
      ])
      setQuestion('')
      setError(null)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
      void vars
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : '问答失败'),
  })

  const handleSubmit = () => {
    const q = question.trim()
    if (!q || queryMutation.isPending) return
    if (selectedSources.length === 0) {
      setError('请至少选择一个知识库文档')
      return
    }
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    queryMutation.mutate({
      question: q,
      model: selectedModel || null,
      sources: selectedSources,
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex h-screen flex-col">
      <div className="border-b border-gray-200 bg-white px-8 py-6">
        <h1 className="text-3xl font-semibold text-gray-900">智能问答</h1>
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="text-sm text-gray-600">
            问答模型
            <select
              value={selectedModel}
              onChange={(e) => setModel(e.target.value)}
              className="ml-2 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <div ref={kbRef} className="relative text-sm text-gray-600">
            知识库
            <button
              type="button"
              onClick={() => setKbOpen((open) => !open)}
              disabled={documents.length === 0}
              className="ml-2 inline-flex items-center gap-1 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 outline-none disabled:cursor-not-allowed disabled:opacity-50"
            >
              {kbLabel}
              <ChevronDown size={14} className="text-gray-400" />
            </button>
            {kbOpen && documents.length > 0 && (
              <div className="absolute left-0 z-20 mt-2 w-80 rounded-xl border border-gray-200 bg-white p-3 shadow-lg">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs text-gray-500">选择检索范围</span>
                  <div className="flex gap-2 text-xs">
                    <button
                      type="button"
                      onClick={selectAllSources}
                      className="text-gray-600 hover:text-gray-900"
                    >
                      全选
                    </button>
                    <button
                      type="button"
                      onClick={clearAllSources}
                      className="text-gray-600 hover:text-gray-900"
                    >
                      清空
                    </button>
                  </div>
                </div>
                <div className="max-h-56 space-y-1 overflow-y-auto">
                  {documents.map((doc) => (
                    <label
                      key={doc.source}
                      className="flex cursor-pointer items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-gray-50"
                    >
                      <input
                        type="checkbox"
                        checked={selectedSources.includes(doc.source)}
                        onChange={() => toggleSource(doc.source)}
                        className="mt-0.5"
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm text-gray-900">{doc.source}</span>
                        <span className="text-xs text-gray-400">{doc.chunk_count} 个片段</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
          {health && (
            <span
              className={`rounded-full px-3 py-1 text-xs ${health.ready ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'}`}
            >
              {health.ready ? '服务就绪' : '服务未就绪'}
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-8 py-6">
            {messages.length === 0 && (
              <div className="flex h-full items-center justify-center text-sm text-gray-400">
                输入问题开始对话，系统将基于知识库检索并生成回答
              </div>
            )}
            <div className="mx-auto max-w-3xl space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 ${
                      msg.role === 'user'
                        ? 'bg-gray-900 text-white'
                        : 'border border-gray-200 bg-white text-gray-800'
                    }`}
                  >
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))}
              {queryMutation.isPending && (
                <div className="text-sm text-gray-400">正在生成回答...</div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          {error && (
            <div className="mx-8 mb-2 rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
              {error}
            </div>
          )}

          <div className="border-t border-gray-200 bg-white px-8 py-4">
            <div className="mx-auto flex max-w-3xl items-end gap-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!health?.ready || queryMutation.isPending || selectedSources.length === 0}
                placeholder="输入你的问题，Shift+Enter 换行"
                rows={2}
                className="flex-1 resize-none rounded-2xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-gray-400 disabled:bg-gray-50"
              />
              <button
                type="button"
                onClick={handleSubmit}
                disabled={
                  !health?.ready ||
                  !question.trim() ||
                  queryMutation.isPending ||
                  selectedSources.length === 0
                }
                className="inline-flex items-center gap-2 rounded-2xl bg-gray-900 px-4 py-3 text-sm text-white disabled:opacity-50"
              >
                <Send size={16} />
                发送
              </button>
            </div>
          </div>
        </div>

        <aside className="w-80 shrink-0 overflow-y-auto border-l border-gray-200 bg-white px-4 py-6">
          <h2 className="mb-4 text-sm font-medium text-gray-900">
            引用来源 ({latestCitations.length})
          </h2>
          {latestCitations.length === 0 ? (
            <p className="text-sm text-gray-400">暂无引用来源</p>
          ) : (
            <div className="space-y-3">
              {latestCitations.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => setActiveCitation(c.id)}
                  className={`w-full rounded-xl border p-3 text-left text-sm transition ${
                    activeCitation === c.id
                      ? 'border-gray-900 bg-gray-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="rounded bg-gray-100 px-2 py-0.5 text-xs font-medium">
                      [{c.id}]
                    </span>
                    <span className="truncate text-xs text-gray-500">{c.source}</span>
                  </div>
                  {c.distance != null && (
                    <div className="mb-2 text-xs text-gray-400">distance: {c.distance.toFixed(3)}</div>
                  )}
                  <p className="line-clamp-4 text-xs leading-5 text-gray-600">{c.text}</p>
                </button>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
