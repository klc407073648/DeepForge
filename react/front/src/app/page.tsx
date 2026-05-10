import Link from "next/link";
import { ContactForm } from "@/features/contact/contact-form";
import { Dashboard } from "@/features/dashboard/dashboard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <main className="app-shell">
      <div className="app-container">
        <section className="hero-section">
          <div className="hero-copy">
            <div className="hero-badge">DeepForge Starter · Full Stack</div>
            <h1 className="hero-title">Next.js + Payload CMS 全栈模板</h1>
            <p className="hero-description">
              前端内置 shadcn/ui 风格组件、Tailwind CSS、TanStack Query、Zustand、React Hook Form、Zod 与
              Recharts，适合作为中后台、SaaS 控制台或内容管理项目的起点。
            </p>
            <div className="hero-actions">
              <Button asChild>
                <Link href="/login">登录门户</Link>
              </Button>
              <Button variant="outline" asChild>
                <Link href="/register">注册账号</Link>
              </Button>
            </div>
          </div>

          <Card className="hero-panel glass-panel">
            <CardHeader>
              <CardTitle>技术栈</CardTitle>
              <CardDescription>按职责拆分，避免把接口数据放入客户端状态。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="stack-list">
                <div className="stack-item">
                  <span>Next.js App Router</span>
                  <span>Frontend</span>
                </div>
                <div className="stack-item">
                  <span>TanStack Query + fetch</span>
                  <span>Server State</span>
                </div>
                <div className="stack-item">
                  <span>Zustand</span>
                  <span>UI State</span>
                </div>
                <div className="stack-item">
                  <span>Payload CMS + PostgreSQL</span>
                  <span>Backend</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="section-grid">
          <Dashboard />
          <Card className="form-shell">
            <CardHeader>
              <CardTitle>示例表单</CardTitle>
              <CardDescription>使用 React Hook Form + Zod 完成类型安全校验。</CardDescription>
            </CardHeader>
            <CardContent>
              <ContactForm />
            </CardContent>
          </Card>
        </section>
      </div>
    </main>
  );
}
