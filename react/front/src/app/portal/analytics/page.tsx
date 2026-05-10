import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const metrics = [
  {
    label: "浏览量",
    value: "48,920",
    change: "+12.4%"
  },
  {
    label: "内容转化",
    value: "8.7%",
    change: "+2.1%"
  },
  {
    label: "活跃用户",
    value: "3,186",
    change: "+9.8%"
  }
];

export default function PortalAnalyticsPage() {
  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Analytics</p>
        <h2>数据分析</h2>
        <span>观察门户内容效果和用户行为。</span>
      </div>

      <div className="stat-grid">
        {metrics.map((metric) => (
          <Card key={metric.label} className="stat-card">
            <CardHeader>
              <CardDescription>{metric.label}</CardDescription>
              <CardTitle className="text-3xl">{metric.value}</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="metric-up">{metric.change}</span>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card className="analytics-panel">
        <CardHeader>
          <CardTitle>访问趋势</CardTitle>
          <CardDescription>用于后续接入 ECharts 或 Recharts 的分析区域。</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="analytics-bars">
            {[42, 68, 56, 82, 74, 91, 63].map((height, index) => (
              <span key={index} style={{ height: `${height}%` }} />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
