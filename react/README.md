# DeepForge Starter

这是一个按推荐方案拆分的全栈模板：

- `front`: Next.js App Router + TypeScript + shadcn/ui 风格组件 + Tailwind CSS + TanStack Query + Zustand + React Hook Form + Zod + Recharts + Vitest
- `backend`: Payload CMS + PostgreSQL + TypeScript，提供 CMS 后台和 REST API

## 本地启动

先启动 PostgreSQL：

```bash
docker compose up postgres
```

启动后端：

```bash
cd backend
cp .env.example .env
npm install
npm run dev
```

后端地址：

- Payload Admin: `http://localhost:3001/admin`
- REST API: `http://localhost:3001/api`

启动前端：

```bash
cd front
cp .env.example .env.local
npm install
npm run dev
```

前端地址：

- App: `http://localhost:3000`

## Docker 启动

```bash
docker compose up --build
```

Nginx 入口：

- `http://localhost:8080`

如需启用 Redis：

```bash
docker compose --profile cache up redis
```

## 代码规范

两个子项目都内置：

- ESLint
- Prettier
- Husky
- lint-staged

建议提交前执行：

```bash
npm run lint
npm run typecheck
npm test
```

## 目录说明

```txt
front/
  src/app                 Next.js App Router
  src/components/ui       shadcn/ui 风格基础组件
  src/features            业务模块示例
  src/lib                 API 与工具函数
  src/stores              Zustand 状态

backend/
  src/app                 Payload + Next.js 入口
  src/collections         Payload 集合定义
  src/payload.config.ts   Payload 主配置
```

### shadcn/ui

https://www.shadcn.com.cn/docs
