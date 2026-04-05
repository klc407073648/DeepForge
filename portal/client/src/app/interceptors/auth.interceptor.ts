import { Injectable } from '@angular/core';
import {
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor,
  HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError, BehaviorSubject } from 'rxjs';
import { catchError, filter, take, switchMap } from 'rxjs/operators';
import { AuthHttpService } from '../services/auth-http.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  private isRefreshing = false;
  private refreshTokenSubject: BehaviorSubject<any> = new BehaviorSubject<any>(
    null
  );

  constructor(private authService: AuthHttpService) {}

  intercept(
    request: HttpRequest<unknown>,
    next: HttpHandler
  ): Observable<HttpEvent<unknown>> {
    // 为需要认证的请求添加 Authorization 头
    if (this.isAuthRequest(request)) {
      request = this.addTokenHeader(request);
      console.log('[AuthInterceptor] 已添加 Authorization 头到请求:', request.url);
    } else {
      console.log('[AuthInterceptor] 跳过添加 Authorization 头:', request.url);
    }

    return next.handle(request).pipe(
      catchError((error: HttpErrorResponse) => {
        // 如果是 401 错误且不是登录请求，尝试刷新令牌
        if (
          error.status === 401 &&
          !this.isLoginRequest(request) &&
          !this.isRefreshTokenRequest(request)
        ) {
          return this.handle401Error(request, next);
        }

        return throwError(() => error);
      })
    );
  }

  /**
   * 判断是否为需要认证的请求
   */
  private isAuthRequest(request: HttpRequest<any>): boolean {
    const authUrls = ['/api/auth/me', '/api/users', '/api/manage'];
    return authUrls.some((url) => request.url.includes(url));
  }

  /**
   * 判断是否为登录请求
   */
  private isLoginRequest(request: HttpRequest<any>): boolean {
    return request.url.includes('/api/auth/login');
  }

  /**
   * 判断是否为刷新令牌请求
   */
  private isRefreshTokenRequest(request: HttpRequest<any>): boolean {
    return request.url.includes('/api/auth/refresh');
  }

  /**
   * 为请求添加 Authorization 头
   */
  private addTokenHeader(request: HttpRequest<any>): HttpRequest<any> {
    const token = this.authService.getToken();

    if (token) {
      console.log('[AuthInterceptor] Token 已获取，长度:', token.length);
      return request.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`,
        },
      });
    } else {
      console.warn('[AuthInterceptor] 未找到 Token，请先登录');
    }

    return request;
  }

  /**
   * 处理 401 错误（令牌过期）
   */
  private handle401Error(
    request: HttpRequest<any>,
    next: HttpHandler
  ): Observable<HttpEvent<any>> {
    if (!this.isRefreshing) {
      this.isRefreshing = true;
      this.refreshTokenSubject.next(null);

      return this.authService.refreshToken().pipe(
        switchMap((response: any) => {
          this.isRefreshing = false;
          this.refreshTokenSubject.next(response.data?.token);

          // 重新发送原始请求
          return next.handle(this.addTokenHeader(request));
        }),
        catchError((error) => {
          this.isRefreshing = false;
          this.authService.logout().subscribe();
          return throwError(() => error);
        })
      );
    }

    // 如果正在刷新令牌，等待刷新完成
    return this.refreshTokenSubject.pipe(
      filter((token) => token !== null),
      take(1),
      switchMap(() => next.handle(this.addTokenHeader(request)))
    );
  }
}
