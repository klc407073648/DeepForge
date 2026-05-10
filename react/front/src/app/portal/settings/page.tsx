"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuthStore } from "@/stores/auth-store";

export default function PortalSettingsPage() {
  const currentUser = useAuthStore((state) => state.currentUser);

  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Settings</p>
        <h2>账户设置</h2>
        <span>查看当前模拟登录用户信息。</span>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>个人资料</CardTitle>
          <CardDescription>当前版本暂只读，接入真实后端后可保存到 Payload users。</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="settings-form">
            <div className="field-group">
              <label className="field-label" htmlFor="name">
                姓名
              </label>
              <Input id="name" value={currentUser?.name ?? ""} readOnly />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="email">
                邮箱
              </label>
              <Input id="email" value={currentUser?.email ?? ""} readOnly />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="role">
                角色
              </label>
              <Input id="role" value={currentUser?.role ?? ""} readOnly />
            </div>
          </div>
        </CardContent>
      </Card>
      <div className="settings-grid">
        <Card>
          <CardHeader>
            <CardTitle>安全状态</CardTitle>
            <CardDescription>本地模拟认证仅用于前端流程验证。</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="security-card">
              <span>模拟会话</span>
              <strong>已启用</strong>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>后续集成</CardTitle>
            <CardDescription>可替换为 Payload Users 登录注册接口。</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="security-card">
              <span>真实认证</span>
              <strong>待接入</strong>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
