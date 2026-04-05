# Go 认证服务后端

这是一个用Go语言编写的认证服务后端，为Angular前端应用提供用户认证功能。

## 功能特性

- JWT令牌认证
- 用户登录/登出
- 令牌刷新机制
- 角色权限管理（admin/user）
- 密码加密存储
- CORS跨域支持
- SQLite数据库

## 项目结构

```
backend/
├── main.go              # 主程序入口
├── go.mod              # Go模块依赖
├── models/             # 数据模型
│   ├── user.go         # 用户相关模型
│   └── response.go     # API响应模型
├── handlers/           # HTTP处理器
│   └── auth.go         # 认证相关处理器
├── services/           # 业务逻辑服务
│   └── auth.go         # 认证服务
├── middleware/         # 中间件
│   └── auth.go         # 认证中间件
├── utils/              # 工具类
│   └── jwt.go          # JWT工具
└── database/           # 数据库相关
```

## 安装和运行

### 1. 安装Go依赖

```bash
cd backend
go mod tidy
```

### 2. 运行服务

```bash
go run main.go
```

服务将在 `http://localhost:3000` 启动

### 3. 测试API

#### 登录接口
```bash
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

#### 获取用户信息
```bash
curl -X GET http://localhost:3000/api/auth/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## API接口

### 认证相关

- `POST /api/auth/login` - 用户登录
- `POST /api/auth/logout` - 用户登出
- `POST /api/auth/refresh` - 刷新令牌
- `GET /api/auth/me` - 获取当前用户信息

### 请求/响应格式

#### 登录请求
```json
{
  "username": "admin",
  "password": "admin123"
}
```

#### 登录响应
```json
{
  "success": true,
  "message": "登录成功",
  "data": {
    "user": {
      "id": 1,
      "username": "admin",
      "email": "admin@example.com",
      "role": "admin",
      "name": "管理员"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 86400
  }
}
```

## 默认用户

系统会自动创建一个默认管理员用户：
- 用户名: `admin`
- 密码: `admin123`
- 角色: `admin`

## 配置说明

- 数据库: SQLite (auth.db)
- JWT密钥: 在 `utils/jwt.go` 中配置
- 端口: 3000
- CORS: 允许 `http://localhost:4200` (Angular开发服务器)

## 安全注意事项

1. 生产环境中请修改JWT密钥
2. 使用环境变量管理敏感配置
3. 考虑使用更安全的数据库（如PostgreSQL）
4. 实现令牌黑名单机制
5. 添加请求频率限制


