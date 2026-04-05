export enum UserRole {
  Admin = 'admin',
  User = 'user',
}

export interface User {
  id: number;
  username: string;
  password?: string; // 可选，API 响应中通常不包含密码
  role: UserRole;
  avatar?: string; // 头像
  lastLoginAt?: string; // 最后登录时间
  createdAt?: string; // 创建时间
  updatedAt?: string; // 更新时间
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  password: string;
}
