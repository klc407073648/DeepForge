import {
  ApplicationConfig,
  provideZoneChangeDetection,
  importProvidersFrom,
  APP_INITIALIZER,
} from '@angular/core';
import { provideRouter } from '@angular/router';
import { HttpClient } from '@angular/common/http';

import { routes } from './app.routes';
import {
  provideClientHydration,
  withEventReplay,
} from '@angular/platform-browser';
import { provideNzI18n, zh_CN } from 'ng-zorro-antd/i18n';
import { registerLocaleData } from '@angular/common';
import zh from '@angular/common/locales/zh';
import en from '@angular/common/locales/en';
import { FormsModule } from '@angular/forms';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import {
  provideHttpClient,
  withInterceptorsFromDi,
} from '@angular/common/http';
import { HTTP_INTERCEPTORS } from '@angular/common/http';
import { AuthInterceptor } from './interceptors/auth.interceptor';
import { ErrorInterceptor } from './interceptors/error.interceptor';
import {
  TranslateModule,
  TranslateLoader,
  TranslateService,
} from '@ngx-translate/core';
import { HttpClient as NgHttpClient } from '@angular/common/http';
import { QiankunService } from './services/qiankun.service';

// 注册语言数据
registerLocaleData(zh);
registerLocaleData(en);

// HTTP loader factory for ngx-translate
export function createTranslateLoader(http: NgHttpClient): TranslateLoader {
  return {
    getTranslation(lang: string) {
      return http.get<any>(`/assets/i18n/${lang}.json`);
    },
  };
}

// 初始化翻译
export function initializeTranslation(translate: TranslateService) {
  return () => {
    translate.setDefaultLang('zh');
    return translate.use('zh').toPromise();
  };
}

// 初始化qiankun
export function initializeQiankun(qiankun: QiankunService) {
  return () => {
    console.log('APP_INITIALIZER: Initializing qiankun service');
    // qiankun服务在构造函数中已经初始化，这里只是确保服务被创建
    return Promise.resolve();
  };
}


export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideClientHydration(withEventReplay()),
    provideNzI18n(zh_CN),
    importProvidersFrom(
      FormsModule,
      TranslateModule.forRoot({
        defaultLanguage: 'zh',
        loader: {
          provide: TranslateLoader,
          useFactory: createTranslateLoader,
          deps: [NgHttpClient],
        },
      })
    ),
    provideAnimationsAsync(),
    provideHttpClient(withInterceptorsFromDi()),
    // 初始化翻译服务
    {
      provide: APP_INITIALIZER,
      useFactory: initializeTranslation,
      deps: [TranslateService],
      multi: true,
    },
    // 初始化qiankun服务
    {
      provide: APP_INITIALIZER,
      useFactory: initializeQiankun,
      deps: [QiankunService],
      multi: true,
    },
    // 注册 HTTP 拦截器
    {
      provide: HTTP_INTERCEPTORS,
      useClass: AuthInterceptor,
      multi: true,
    },
    {
      provide: HTTP_INTERCEPTORS,
      useClass: ErrorInterceptor,
      multi: true,
    },
  ],
};
