import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Pencil, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { deleteKey, getKeys, upsertKey } from '../api/rag'
import { ApiError } from '../api/client'
import type { KeyConfigItem } from '../types/api'

export default function ApiKeyPage() {
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState<KeyConfigItem | null>(null)
  const [value, setValue] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { data, isLoading } = useQuery({ queryKey: ['keys'], queryFn: getKeys })

  const saveMutation = useMutation({
    mutationFn: ({ name, val }: { name: string; val: string }) => upsertKey(name, val),
    onSuccess: () => {
      setMessage('配置已保存')
      setEditing(null)
      setValue('')
      queryClient.invalidateQueries({ queryKey: ['keys'] })
      queryClient.invalidateQueries({ queryKey: ['health'] })
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : '保存失败'),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteKey,
    onSuccess: () => {
      setMessage('运行时覆盖已清除，将回退到 .env 配置')
      queryClient.invalidateQueries({ queryKey: ['keys'] })
      queryClient.invalidateQueries({ queryKey: ['health'] })
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : '删除失败'),
  })

  const items = data?.items ?? []

  return (
    <div className="mx-auto max-w-5xl px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900">API Key 管理</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-500">
          列表中展示的是当前生效配置的脱敏值。在此页面的修改会写入运行时配置（data/runtime_config.json），
          优先级高于 .env。请妥善保管 Key，不要分享给他人。
        </p>
      </div>

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

      <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-100 bg-gray-50 text-gray-500">
            <tr>
              <th className="px-4 py-3 font-medium">名称</th>
              <th className="px-4 py-3 font-medium">Key / 值</th>
              <th className="px-4 py-3 font-medium">用途</th>
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
            {items.map((item) => (
              <tr key={item.name} className="border-b border-gray-50">
                <td className="px-4 py-3 font-medium text-gray-900">{item.label}</td>
                <td className="px-4 py-3 font-mono text-xs text-gray-600">
                  {item.masked_value || '—'}
                </td>
                <td className="px-4 py-3 text-gray-600">{item.purpose}</td>
                <td className="px-4 py-3">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${
                      item.configured
                        ? 'bg-green-50 text-green-700'
                        : 'bg-gray-100 text-gray-500'
                    }`}
                  >
                    {item.configured ? '已配置' : '未配置'}
                  </span>
                  {item.source === 'runtime' && (
                    <span className="ml-2 text-xs text-gray-400">运行时</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setEditing(item)
                        setValue('')
                      }}
                      className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100"
                    >
                      <Pencil size={16} />
                    </button>
                    {item.source === 'runtime' && (
                      <button
                        type="button"
                        onClick={() => {
                          if (window.confirm(`清除 ${item.label} 的运行时覆盖？`)) {
                            deleteMutation.mutate(item.name)
                          }
                        }}
                        className="rounded-lg p-1.5 text-gray-500 hover:bg-red-50 hover:text-red-600"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {editing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
            <h3 className="mb-2 text-lg font-medium">编辑 {editing.label}</h3>
            <p className="mb-4 text-xs text-gray-500">留空并保存将清除运行时覆盖</p>
            <input
              type="password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="输入新值"
              className="w-full rounded-xl border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
            />
            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => {
                  setEditing(null)
                  setValue('')
                }}
                className="rounded-xl px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
              >
                取消
              </button>
              <button
                type="button"
                disabled={saveMutation.isPending}
                onClick={() => saveMutation.mutate({ name: editing.name, val: value })}
                className="rounded-xl bg-gray-900 px-4 py-2 text-sm text-white disabled:opacity-50"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
