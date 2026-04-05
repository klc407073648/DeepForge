import { Injectable } from '@angular/core';
import {
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpInterceptor,
  HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

@Injectable()
export class ErrorInterceptor implements HttpInterceptor {
  constructor() {}

  intercept(
    request: HttpRequest<unknown>,
    next: HttpHandler
  ): Observable<HttpEvent<unknown>> {
    return next.handle(request).pipe(
      catchError((error: HttpErrorResponse) => {
        let errorMessage = '发生未知错误';

        if (error.error instanceof ErrorEvent) {
          // 客户端错误
          errorMessage = `客户端错误: ${error.error.message}`;
        } else {
          // 服务器错误
          switch (error.status) {
            case 400:
              errorMessage = error.error?.message || '请求参数错误';
              break;
            case 401:
              errorMessage = '未授权，请重新登录';
              break;
            case 403:
              errorMessage = '权限不足';
              break;
            case 404:
              errorMessage = '请求的资源不存在';
              break;
            case 409:
              errorMessage = '数据冲突';
              break;
            case 422:
              errorMessage = '数据验证失败';
              break;
            case 429:
              errorMessage = '请求过于频繁，请稍后再试';
              break;
            case 500:
              errorMessage = '服务器内部错误';
              break;
            case 502:
              errorMessage = '网关错误';
              break;
            case 503:
              errorMessage = '服务暂时不可用';
              break;
            case 504:
              errorMessage = '网关超时';
              break;
            default:
              errorMessage =
                error.error?.message || `HTTP ${error.status} 错误`;
          }
        }

        // 可以在这里添加全局错误通知
        console.error('HTTP 错误:', {
          url: request.url,
          method: request.method,
          status: error.status,
          message: errorMessage,
          error: error.error,
        });

        // 可以集成消息服务显示错误提示
        // this.messageService.error(errorMessage);

        return throwError(() => new Error(errorMessage));
      })
    );
  }
}

