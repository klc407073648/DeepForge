import { Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, throwError, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import {
  LoginRequest,
  RegisterRequest,
  User,
  UserRole,
} from '../model/user.model';
import {
  ApiResponse,
  LoginResponse,
  UserInfoResponse,
  RefreshTokenRequest,
  RefreshTokenResponse,
  UserListRequest,
  UserListResponse,
  UpdateUserRoleRequest
} from '../model/api.model';

@Injectable({
  providedIn: 'root',
})
export class AuthHttpService {
  private readonly API_BASE_URL = 'http://localhost:3000/api'; // 根据实际后端地址修改
  private readonly TOKEN_KEY = 'auth_token';
  private readonly REFRESH_TOKEN_KEY = 'refresh_token';
  private readonly USER_KEY = 'current_user';

  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  private tokenSubject = new BehaviorSubject<string | null>(null);
  public token$ = this.tokenSubject.asObservable();

  constructor(
    @Inject(PLATFORM_ID) private platformId: Object,
    private http: HttpClient
  ) {
    this.initializeAuth();
  }

  /**
   * 初始化认证状态
   */
  private initializeAuth(): void {
    if (isPlatformBrowser(this.platformId)) {
      const token = localStorage.getItem(this.TOKEN_KEY);
      const user = localStorage.getItem(this.USER_KEY);

      if (token && user) {
        this.tokenSubject.next(token);
        this.currentUserSubject.next(JSON.parse(user));
      }
    }
  }

  /**
   * 用户登录
   */
  login(credentials: LoginRequest): Observable<ApiResponse<LoginResponse>> {
    return this.http
      .post<
        ApiResponse<LoginResponse>
      >(`${this.API_BASE_URL}/auth/login`, credentials)
      .pipe(
        tap((response) => {
          if (response.success && response.data) {
            this.setAuthData(response.data);
          }
        }),
        catchError(this.handleError)
      );
  }

  /**
   * 用户注册
   */
  register(
    registerData: RegisterRequest
  ): Observable<ApiResponse<LoginResponse>> {
    return this.http
      .post<
        ApiResponse<LoginResponse>
      >(`${this.API_BASE_URL}/auth/register`, registerData)
      .pipe(
        tap((response) => {
          if (response.success && response.data) {
            this.setAuthData(response.data);
          }
        }),
        catchError(this.handleError)
      );
  }

  /**
   * 用户登出
   */
  logout(): Observable<ApiResponse> {
    const refreshToken = this.getRefreshToken();

    if (refreshToken) {
      return this.http
        .post<ApiResponse>(`${this.API_BASE_URL}/auth/logout`, { refreshToken })
        .pipe(
          tap(() => this.clearAuthData()),
          catchError(() => {
            // 即使服务器登出失败，也要清除本地数据
            this.clearAuthData();
            return of({ success: true, message: '已登出' });
          })
        );
    } else {
      this.clearAuthData();
      return of({ success: true, message: '已登出' });
    }
  }

  /**
   * 获取当前用户信息
   */
  getCurrentUserInfo(): Observable<ApiResponse<UserInfoResponse>> {
    return this.http
      .get<ApiResponse<UserInfoResponse>>(`${this.API_BASE_URL}/auth/me`)
      .pipe(
        tap((response) => {
          if (response.success && response.data) {
            this.currentUserSubject.next(response.data.user);
            this.saveUserToStorage(response.data.user);
          }
        }),
        catchError(this.handleError)
      );
  }

  /**
   * 获取用户列表
   */
  getUserList(params: UserListRequest): Observable<ApiResponse<UserListResponse>> {
    return this.http.get<ApiResponse<UserListResponse>>(
      `${this.API_BASE_URL}/users/list`,
      { params: params as any }
    ).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * 更新用户角色
   */
  updateUserRole(request: UpdateUserRoleRequest): Observable<ApiResponse> {
    return this.http.put<ApiResponse>(
      `${this.API_BASE_URL}/users/role`,
      request
    ).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * 刷新访问令牌
   */
  refreshToken(): Observable<ApiResponse<RefreshTokenResponse>> {
    const refreshToken = this.getRefreshToken();

    if (!refreshToken) {
      return throwError(() => new Error('没有刷新令牌'));
    }

    const request: RefreshTokenRequest = { refreshToken };

    return this.http
      .post<
        ApiResponse<RefreshTokenResponse>
      >(`${this.API_BASE_URL}/auth/refresh`, request)
      .pipe(
        tap((response) => {
          if (response.success && response.data) {
            this.setToken(response.data.token);
            this.setRefreshToken(response.data.refreshToken);
          }
        }),
        catchError(this.handleError)
      );
  }

  /**
   * 检查用户是否已登录
   */
  isLoggedIn(): boolean {
    return this.currentUserSubject.value !== null && this.getToken() !== null;
  }

  /**
   * 获取当前用户
   */
  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * 检查是否为管理员
   */
  isAdmin(): boolean {
    return this.currentUserSubject.value?.role === UserRole.Admin;
  }

  /**
   * 检查是否为普通用户
   */
  isUser(): boolean {
    return this.currentUserSubject.value?.role === UserRole.User;
  }

  /**
   * 检查用户权限
   */
  hasPermission(permission: string): boolean {
    const user = this.getCurrentUser();
    if (!user) return false;

    // 管理员拥有所有权限
    if (user.role === UserRole.Admin) return true;

    // 这里可以扩展为从服务器获取权限列表
    // 目前使用本地权限配置
    const userPermissions = this.getUserPermissions(user.role);
    return userPermissions.includes(permission);
  }

  /**
   * 获取访问令牌
   */
  getToken(): string | null {
    if (isPlatformBrowser(this.platformId)) {
      return localStorage.getItem(this.TOKEN_KEY);
    }
    return null;
  }

  /**
   * 获取刷新令牌
   */
  private getRefreshToken(): string | null {
    if (isPlatformBrowser(this.platformId)) {
      return localStorage.getItem(this.REFRESH_TOKEN_KEY);
    }
    return null;
  }

  /**
   * 设置认证数据
   */
  private setAuthData(loginData: LoginResponse): void {
    this.setToken(loginData.token);
    if (loginData.refreshToken) {
      this.setRefreshToken(loginData.refreshToken);
    }
    this.currentUserSubject.next(loginData.user);
    this.saveUserToStorage(loginData.user);
  }

  /**
   * 设置访问令牌
   */
  private setToken(token: string): void {
    this.tokenSubject.next(token);
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem(this.TOKEN_KEY, token);
    }
  }

  /**
   * 设置刷新令牌
   */
  private setRefreshToken(refreshToken: string): void {
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem(this.REFRESH_TOKEN_KEY, refreshToken);
    }
  }

  /**
   * 保存用户信息到本地存储
   */
  private saveUserToStorage(user: User): void {
    if (isPlatformBrowser(this.platformId)) {
      localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    }
  }

  /**
   * 清除认证数据
   */
  private clearAuthData(): void {
    this.currentUserSubject.next(null);
    this.tokenSubject.next(null);

    if (isPlatformBrowser(this.platformId)) {
      localStorage.removeItem(this.TOKEN_KEY);
      localStorage.removeItem(this.REFRESH_TOKEN_KEY);
      localStorage.removeItem(this.USER_KEY);
    }
  }

  /**
   * 获取用户权限列表
   */
  private getUserPermissions(role: string): string[] {
    const permissions: { [key: string]: string[] } = {
      admin: ['*'], // 管理员拥有所有权限
      user: ['manage.Directives.view', 'welcome.view'],
    };
    return permissions[role] || [];
  }

  /**
   * 错误处理
   */
  private handleError = (error: HttpErrorResponse): Observable<never> => {
    let errorMessage = '发生未知错误';

    if (error.error instanceof ErrorEvent) {
      // 客户端错误
      errorMessage = `客户端错误: ${error.error.message}`;
    } else {
      // 服务器错误
      if (error.status === 401) {
        errorMessage = '未授权，请重新登录';
        this.clearAuthData();
      } else if (error.status === 403) {
        errorMessage = '权限不足';
      } else if (error.status === 404) {
        errorMessage = '请求的资源不存在';
      } else if (error.status >= 500) {
        errorMessage = '服务器内部错误';
      } else if (error.error?.message) {
        errorMessage = error.error.message;
      }
    }

    console.error('认证服务错误:', error);
    return throwError(() => new Error(errorMessage));
  };
}
