import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import ApiKeyPage from './pages/ApiKeyPage'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Navigate to="/knowledge" replace />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/settings/keys" element={<ApiKeyPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
