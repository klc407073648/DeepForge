import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const users = [
  {
    name: "管理员",
    email: "admin@deepforge.dev",
    role: "Admin",
    status: "在线"
  },
  {
    name: "内容编辑",
    email: "editor@deepforge.dev",
    role: "Editor",
    status: "活跃"
  },
  {
    name: "运营成员",
    email: "ops@deepforge.dev",
    role: "Editor",
    status: "离线"
  }
];

export default function PortalUsersPage() {
  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Users</p>
        <h2>用户管理</h2>
        <span>查看门户成员和内容权限。</span>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>成员列表</CardTitle>
          <CardDescription>当前为前端模拟数据，后续可对接 Payload users 集合。</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="user-list">
            {users.map((user) => (
              <div key={user.email} className="user-list-item">
                <div className="user-avatar">{user.name.slice(0, 1)}</div>
                <div>
                  <strong>{user.name}</strong>
                  <span>{user.email}</span>
                </div>
                <div className="user-tags">
                  <em>{user.role}</em>
                  <span>{user.status}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
