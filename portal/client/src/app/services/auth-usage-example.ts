/**
 * 认证服务使用示例
 * 这个文件展示了如何在组件中使用 AuthHttpService
 */

import { Component, OnInit, OnDestroy, Injectable } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, CanActivate } from '@angular/router';
import { Subject, takeUntil, Observable } from 'rxjs';
import { map, take } from 'rxjs/operators';
import { AuthHttpService } from './auth-http.service';
import { User } from '../model/user.model';

@Component({
  selector: 'app-example',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div>
      <h2>认证服务使用示例</h2>

      <!-- 用户信息显示 -->
      <div *ngIf="currentUser; else loginPrompt">
        <h3>欢迎, {{ currentUser.username }}!</h3>
        <p>角色: {{ currentUser.role }}</p>

        <!-- 权限检查示例 -->
        <div *ngIf="authService.isAdmin()">
          <p style="color: green;">您拥有管理员权限</p>
        </div>

        <div *ngIf="authService.hasPermission('manage.Directives.view')">
          <p style="color: blue;">您有查看管理列表的权限</p>
        </div>

        <button (click)="logout()">登出</button>
      </div>

      <ng-template #loginPrompt>
        <p>请先登录</p>
        <button (click)="goToLogin()">去登录</button>
      </ng-template>
    </div>
  `,
})
export class AuthUsageExampleComponent implements OnInit, OnDestroy {
  currentUser: User | null = null;
  private destroy$ = new Subject<void>();

  constructor(
    public authService: AuthHttpService,
    private router: Router
  ) {}

  ngOnInit(): void {
    // 订阅用户状态变化
    this.authService.currentUser$
      .pipe(takeUntil(this.destroy$))
      .subscribe((user) => {
        this.currentUser = user;
        console.log('用户状态变化:', user);
      });

    // 如果用户已登录，获取最新的用户信息
    if (this.authService.isLoggedIn()) {
      this.refreshUserInfo();
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * 刷新用户信息
   */
  refreshUserInfo(): void {
    this.authService
      .getCurrentUserInfo()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (response.success) {
            console.log('用户信息更新成功:', response.data?.user);
          }
        },
        error: (error) => {
          console.error('获取用户信息失败:', error.message);
          // 如果获取用户信息失败，可能是令牌过期，自动登出
          this.authService.logout().subscribe();
        },
      });
  }

  /**
   * 用户登出
   */
  logout(): void {
    this.authService
      .logout()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (response.success) {
            console.log('登出成功');
            this.router.navigate(['/login']);
          }
        },
        error: (error) => {
          console.error('登出失败:', error.message);
          // 即使登出失败，也跳转到登录页
          this.router.navigate(['/login']);
        },
      });
  }

  /**
   * 跳转到登录页
   */
  goToLogin(): void {
    this.router.navigate(['/login']);
  }

  /**
   * 权限检查示例
   */
  checkPermissions(): void {
    const permissions = [
      'manage.Directives.view',
      'manage.Directives.edit',
      'manage.Directives.delete',
      'admin.users.manage',
    ];

    permissions.forEach((permission) => {
      const hasPermission = this.authService.hasPermission(permission);
      console.log(`权限 ${permission}: ${hasPermission ? '有' : '无'}`);
    });
  }
}

/**
 * 路由守卫使用示例
 */

@Injectable({
  providedIn: 'root',
})
export class AuthGuardExample implements CanActivate {
  constructor(
    private authService: AuthHttpService,
    private router: Router
  ) {}

  canActivate(): Observable<boolean> {
    return this.authService.currentUser$.pipe(
      take(1),
      map((user) => {
        if (user) {
          return true;
        } else {
          this.router.navigate(['/login']);
          return false;
        }
      })
    );
  }
}

/**
 * 管理员守卫使用示例
 */
@Injectable({
  providedIn: 'root',
})
export class AdminGuardExample implements CanActivate {
  constructor(
    private authService: AuthHttpService,
    private router: Router
  ) {}

  canActivate(): Observable<boolean> {
    return this.authService.currentUser$.pipe(
      take(1),
      map((user) => {
        if (user && this.authService.isAdmin()) {
          return true;
        } else {
          this.router.navigate(['/unauthorized']);
          return false;
        }
      })
    );
  }
}

/**
 * HTTP 拦截器使用说明
 *
 * 1. AuthInterceptor 会自动为需要认证的请求添加 Authorization 头
 * 2. 当收到 401 错误时，会自动尝试刷新令牌
 * 3. 如果刷新令牌失败，会自动登出用户
 *
 * 使用方式：
 * - 拦截器已在 app.config.ts 中注册
 * - 无需在组件中手动处理令牌
 * - 所有 HTTP 请求都会自动应用拦截器
 * - 专为浏览器环境优化，直接使用 localStorage
 */
