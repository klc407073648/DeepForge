"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getDashboardStats, getRevenueSeries } from "./api";
import { DashboardChart } from "./dashboard-chart";

const fallbackRevenue = [
  { month: "1月", revenue: 12000 },
  { month: "2月", revenue: 18000 },
  { month: "3月", revenue: 15000 },
  { month: "4月", revenue: 24000 },
  { month: "5月", revenue: 32000 }
];

export function Dashboard() {
  const statsQuery = useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: getDashboardStats
  });

  const revenueQuery = useQuery({
    queryKey: ["dashboard", "revenue"],
    queryFn: getRevenueSeries
  });

  const stats = statsQuery.data ?? {
    users: 1280,
    revenue: 101000,
    conversionRate: 12.8
  };

  return (
    <section className="grid gap-6">
      <div className="stat-grid">
        <Card className="stat-card">
          <CardHeader>
            <CardDescription>用户数</CardDescription>
            <CardTitle className="text-3xl">{stats.users.toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="stat-card">
          <CardHeader>
            <CardDescription>收入</CardDescription>
            <CardTitle className="text-3xl">¥{stats.revenue.toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="stat-card">
          <CardHeader>
            <CardDescription>转化率</CardDescription>
            <CardTitle className="text-3xl">{stats.conversionRate}%</CardTitle>
          </CardHeader>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>收入趋势</CardTitle>
          <CardDescription>
            {statsQuery.isFetching || revenueQuery.isFetching ? "正在同步后端数据..." : "后端未启动时使用本地示例数据。"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DashboardChart data={revenueQuery.data ?? fallbackRevenue} />
        </CardContent>
      </Card>
    </section>
  );
}
