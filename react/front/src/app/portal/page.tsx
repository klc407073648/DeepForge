import { Dashboard } from "@/features/dashboard/dashboard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const quickLinks = [
  {
    title: "待审核内容",
    value: "12",
    description: "3 篇文章需要今天处理"
  },
  {
    title: "素材容量",
    value: "68%",
    description: "媒体库仍有充足空间"
  },
  {
    title: "成员活跃",
    value: "24",
    description: "近 7 天登录成员"
  }
];

export default function PortalDashboardPage() {
  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Dashboard</p>
        <h2>门户仪表盘</h2>
        <span>查看内容、用户和收入概览。</span>
      </div>
      <div className="portal-highlight-grid">
        {quickLinks.map((item) => (
          <Card key={item.title} className="portal-highlight-card">
            <CardHeader>
              <CardDescription>{item.title}</CardDescription>
              <CardTitle>{item.value}</CardTitle>
            </CardHeader>
            <CardContent>
              <span>{item.description}</span>
            </CardContent>
          </Card>
        ))}
      </div>
      <Dashboard />
    </div>
  );
}
