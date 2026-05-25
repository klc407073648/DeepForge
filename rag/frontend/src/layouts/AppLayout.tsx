import { NavLink, Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BookOpen, KeyRound, MessageSquare } from 'lucide-react'
import { getHealth } from '../api/rag'

const navItems = [
  { to: '/knowledge', label: '知识库', icon: BookOpen },
  { to: '/chat', label: '智能问答', icon: MessageSquare },
  { to: '/settings/keys', label: 'API Key 管理', icon: KeyRound },
]

export default function AppLayout() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000,
  })

  return (
    <div className="flex min-h-screen bg-[#f5f5f5]">
      <aside className="flex w-60 shrink-0 flex-col border-r border-gray-200 bg-[#f0f0f0] px-4 py-6">
        <div className="mb-8 px-2">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold text-gray-900">RAG 知识库</span>
            <span className="rounded-full bg-gray-900 px-2 py-0.5 text-xs text-white">Beta</span>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                [
                  'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition',
                  isActive
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:bg-white/70 hover:text-gray-900',
                ].join(' ')
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto space-y-2 px-2 text-xs text-gray-500">
          {health && (
            <div className="flex items-center gap-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${health.ready ? 'bg-green-500' : 'bg-amber-500'}`}
              />
              {health.ready ? '服务就绪' : '服务未就绪'}
            </div>
          )}
          <div>v{health?.version ?? '0.1.0'}</div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
