"use client";

import { BarChart3, FileText, Home, ImageIcon, LogOut, Settings, Users } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/auth-store";

const navItems = [
  {
    title: "仪表盘",
    href: "/portal",
    icon: Home
  },
  {
    title: "文章管理",
    href: "/portal/posts",
    icon: FileText
  },
  {
    title: "媒体管理",
    href: "/portal/media",
    icon: ImageIcon
  },
  {
    title: "用户管理",
    href: "/portal/users",
    icon: Users
  },
  {
    title: "数据分析",
    href: "/portal/analytics",
    icon: BarChart3
  },
  {
    title: "账户设置",
    href: "/portal/settings",
    icon: Settings
  }
];

export function PortalShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const currentUser = useAuthStore((state) => state.currentUser);
  const hydrated = useAuthStore((state) => state.hydrated);
  const logout = useAuthStore((state) => state.logout);

  useEffect(() => {
    if (hydrated && !currentUser) {
      router.replace("/login");
    }
  }, [currentUser, hydrated, router]);

  if (!hydrated) {
    return <main className="portal-loading">正在加载门户...</main>;
  }

  if (!currentUser) {
    return <main className="portal-loading">正在跳转登录...</main>;
  }

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  return (
    <main className="portal-shell">
      <aside className="portal-sidebar">
        <Link href="/" className="portal-brand">
          <span>DF</span>
          <div>
            <strong>DeepForge</strong>
            <small>Content Portal</small>
          </div>
        </Link>

        <nav className="portal-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;

            return (
              <Link key={item.href} href={item.href} className={cn("portal-nav-item", active && "active")}>
                <Icon size={18} />
                {item.title}
              </Link>
            );
          })}
        </nav>

        <div className="portal-user-card">
          <div>
            <p>{currentUser.name}</p>
            <span>{currentUser.email}</span>
          </div>
          <Button variant="ghost" size="sm" onClick={handleLogout}>
            <LogOut size={16} />
            退出
          </Button>
        </div>
      </aside>

      <section className="portal-main">
        <header className="portal-header">
          <div>
            <p>欢迎回来</p>
            <h1>{currentUser.name} 的内容门户</h1>
          </div>
          <Button variant="outline" asChild>
            <Link href="/portal/posts">新建内容</Link>
          </Button>
        </header>
        <div className="portal-content">{children}</div>
      </section>
    </main>
  );
}
