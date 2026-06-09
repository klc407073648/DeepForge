import { describe, expect, it, beforeEach } from 'vitest'
import userEvent from '@testing-library/user-event'
import { screen, within } from '@testing-library/react'
import App from './App'
import { renderWithQueryClient } from '../test/utils/render'

describe('App', () => {
  beforeEach(() => {
    window.history.pushState({}, 'Test page', '/')
  })

  it('默认重定向到知识库页', async () => {
    renderWithQueryClient(<App />)

    expect(await screen.findByRole('heading', { name: '知识库' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '知识库' })).toHaveClass('bg-white')
  })

  it('可导航到智能问答页', async () => {
    const user = userEvent.setup()
    renderWithQueryClient(<App />)

    await screen.findByRole('heading', { name: '知识库' })
    await user.click(screen.getByRole('link', { name: '智能问答' }))

    expect(await screen.findByRole('heading', { name: '智能问答' })).toBeInTheDocument()
  })

  it('可导航到 API Key 管理页', async () => {
    const user = userEvent.setup()
    renderWithQueryClient(<App />)

    await screen.findByRole('heading', { name: '知识库' })
    await user.click(screen.getByRole('link', { name: 'API Key 管理' }))

    expect(await screen.findByRole('heading', { name: 'API Key 管理' })).toBeInTheDocument()
  })

  it('侧边栏展示服务就绪状态', async () => {
    renderWithQueryClient(<App />)

    await screen.findByRole('heading', { name: '知识库' })

    const aside = document.querySelector('aside')!
    expect(await within(aside).findByText('服务就绪')).toBeInTheDocument()
    expect(within(aside).getByText('v0.1.0')).toBeInTheDocument()
  })
})
