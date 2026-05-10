import { api } from "@/lib/api";

export type DashboardStats = {
  users: number;
  revenue: number;
  conversionRate: number;
};

export type RevenuePoint = {
  month: string;
  revenue: number;
};

export async function getDashboardStats() {
  return api<DashboardStats>("/api/dashboard/stats");
}

export async function getRevenueSeries() {
  return api<RevenuePoint[]>("/api/dashboard/revenue");
}
