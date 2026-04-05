import { Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

export interface MicroAppConfig {
  name: string;
  entry: string;
  container: string;
  activeRule: string;
  props?: any;
}

export interface MicroAppState {
  [key: string]: any;
}

// 动态导入 qiankun 相关类型，避免 SSR 环境下的问题
let qiankunModule: any = null;
let MicroApp: any = null;
let MicroAppStateActions: any = null;

@Injectable({
  providedIn: 'root'
})
export class QiankunService {
  private microApps: Map<string, any> = new Map();
  private globalState: any = null;
  private isStarted = false;
  private qiankunInitialized = false;

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {
    // 不在构造函数中初始化，而是在需要时动态初始化
  }

  /**
   * 动态加载 qiankun 模块
   */
  private async loadQiankunModule(): Promise<boolean> {
    if (qiankunModule) {
      return true;
    }

    if (!isPlatformBrowser(this.platformId)) {
      console.warn('[QiankunService] Not in browser environment, skipping qiankun initialization');
      return false;
    }

    try {
      console.log('[QiankunService] Loading qiankun module...');
      qiankunModule = await import('qiankun');
      MicroApp = qiankunModule.MicroApp;
      MicroAppStateActions = qiankunModule.MicroAppStateActions;
      console.log('[QiankunService] Qiankun module loaded successfully');
      return true;
    } catch (error) {
      console.error('[QiankunService] Failed to load qiankun module:', error);
      return false;
    }
  }

  /**
   * 初始化 qiankun
   */
  private async initializeQiankun(): Promise<boolean> {
    if (this.qiankunInitialized) {
      return true;
    }

    if (!isPlatformBrowser(this.platformId)) {
      console.warn('[QiankunService] Not in browser environment, skipping qiankun initialization');
      return false;
    }

    try {
      console.log('[QiankunService] Initializing qiankun...');
      
      // 动态加载 qiankun 模块
      const moduleLoaded = await this.loadQiankunModule();
      if (!moduleLoaded) {
        return false;
      }
      
      // 注册微前端应用
      await this.registerMicroApps();
      
      // 初始化全局状态
      this.initGlobalState();
      
      // 启动 qiankun
      this.startQiankun();
      
      this.qiankunInitialized = true;
      console.log('[QiankunService] Qiankun initialized successfully');
      return true;
    } catch (error) {
      console.error('[QiankunService] Failed to initialize qiankun:', error);
      return false;
    }
  }

  /**
   * 注册微前端应用
   */
  private async registerMicroApps(): Promise<void> {
    if (!qiankunModule || !isPlatformBrowser(this.platformId)) {
      return;
    }

    const microApps: MicroAppConfig[] = [
      {
        name: 'vue-settings-app',
        entry: '//localhost:8081',
        container: '#vue-settings-container',
        activeRule: '/vue-settings',
        props: {
          data: {
            message: 'Hello from Angular main app!',
            currentPath: window.location.pathname
          }
        }
      }
    ];

    qiankunModule.registerMicroApps(microApps, {
      beforeLoad: (app: any) => {
        console.log('[QiankunService] Before load app:', app.name);
        return Promise.resolve();
      },
      beforeMount: (app: any) => {
        console.log('[QiankunService] Before mount app:', app.name);
        return Promise.resolve();
      },
      afterMount: (app: any) => {
        console.log('[QiankunService] After mount app:', app.name);
        return Promise.resolve();
      },
      beforeUnmount: (app: any) => {
        console.log('[QiankunService] Before unmount app:', app.name);
        return Promise.resolve();
      },
      afterUnmount: (app: any) => {
        console.log('[QiankunService] After unmount app:', app.name);
        return Promise.resolve();
      }
    });
  }

  /**
   * 初始化全局状态
   */
  private initGlobalState(): void {
    if (!qiankunModule || !isPlatformBrowser(this.platformId)) {
      return;
    }

    const initialState: MicroAppState = {
      user: null,
      theme: 'light',
      language: 'zh'
    };

    this.globalState = qiankunModule.initGlobalState(initialState);
    
    // 监听全局状态变化
    this.globalState.onGlobalStateChange((state: any, prev: any) => {
      console.log('[QiankunService] Global state changed:', state, prev);
    }, true);
  }

  /**
   * 启动 qiankun
   */
  private startQiankun(): void {
    if (this.isStarted || !qiankunModule || !isPlatformBrowser(this.platformId)) {
      console.warn('[QiankunService] Qiankun already started or not in browser environment');
      return;
    }

    qiankunModule.start({
      sandbox: {
        strictStyleIsolation: false,  // 关闭严格样式隔离，允许Vue样式正常显示
        experimentalStyleIsolation: false  // 关闭实验性样式隔离
      },
      prefetch: 'all',
      singular: false,
      fetch: (url: string, options: any) => {
        console.log('[QiankunService] Fetching:', url);
        return window.fetch(url, options);
      }
    });

    this.isStarted = true;
    console.log('[QiankunService] Qiankun started');
  }

  /**
   * 手动加载 Vue 设置应用
   */
  async loadVueSettingsApp(container: string): Promise<boolean> {
    if (!isPlatformBrowser(this.platformId)) {
      console.warn('[QiankunService] Not in browser environment');
      return false;
    }

    try {
      // 确保 qiankun 已初始化
      const initialized = await this.initializeQiankun();
      if (!initialized) {
        console.error('[QiankunService] Failed to initialize qiankun');
        return false;
      }

      console.log('[QiankunService] Loading Vue settings app into container:', container);
      
      // 检查容器是否存在
      const containerElement = document.querySelector(container);
      if (!containerElement) {
        console.error('[QiankunService] Container not found:', container);
        return false;
      }

      // 如果应用已经加载，先卸载
      if (this.microApps.has('vue-settings-app')) {
        await this.unmountVueSettingsApp();
      }

      // 检查Vue应用是否可访问
      try {
        const response = await fetch('http://localhost:8081', { 
          method: 'HEAD',
          mode: 'no-cors'
        });
        console.log('[QiankunService] Vue app is accessible');
      } catch (error) {
        console.error('[QiankunService] Vue app not accessible:', error);
        return false;
      }

      // 加载微前端应用，添加超时控制
      const loadPromise = qiankunModule.loadMicroApp({
        name: 'vue-settings-app',
        entry: '//localhost:8081',
        container: container,
        props: {
          data: {
            message: 'Hello from Angular main app!',
            currentPath: window.location.pathname
          }
        }
      });

      // 设置10秒超时
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Vue应用加载超时')), 10000);
      });

      const microApp = await Promise.race([loadPromise, timeoutPromise]);
      this.microApps.set('vue-settings-app', microApp);
      
      console.log('[QiankunService] Vue settings app loaded successfully');
      return true;
    } catch (error) {
      console.error('[QiankunService] Failed to load Vue settings app:', error);
      return false;
    }
  }

  /**
   * 卸载 Vue 设置应用
   */
  async unmountVueSettingsApp(): Promise<boolean> {
    try {
      const microApp = this.microApps.get('vue-settings-app');
      if (microApp) {
        console.log('[QiankunService] Unmounting Vue settings app');
        await microApp.unmount();
        this.microApps.delete('vue-settings-app');
        console.log('[QiankunService] Vue settings app unmounted successfully');
        return true;
      }
      return false;
    } catch (error) {
      console.error('[QiankunService] Failed to unmount Vue settings app:', error);
      return false;
    }
  }

  /**
   * 设置全局状态
   */
  setGlobalState(state: Partial<MicroAppState>): boolean {
    if (!this.globalState) {
      console.warn('[QiankunService] Global state not initialized');
      return false;
    }

    try {
      this.globalState.setGlobalState(state);
      console.log('[QiankunService] Global state updated:', state);
      return true;
    } catch (error) {
      console.error('[QiankunService] Failed to set global state:', error);
      return false;
    }
  }

  /**
   * 获取全局状态
   */
  getGlobalState(): MicroAppState | null {
    if (!this.globalState) {
      console.warn('[QiankunService] Global state not initialized');
      return null;
    }

    // qiankun 的 MicroAppStateActions 没有 getGlobalState 方法
    // 我们需要通过其他方式获取状态，这里返回 null 表示无法直接获取
    console.warn('[QiankunService] Cannot directly get global state from qiankun');
    return null;
  }

  /**
   * 检查微前端应用是否已加载
   */
  isMicroAppLoaded(name: string): boolean {
    return this.microApps.has(name);
  }

  /**
   * 获取所有已加载的微前端应用
   */
  getLoadedMicroApps(): string[] {
    return Array.from(this.microApps.keys());
  }

  /**
   * 检查 qiankun 是否已启动
   */
  isQiankunStarted(): boolean {
    return this.isStarted && this.qiankunInitialized;
  }

  /**
   * 重新加载微前端应用
   */
  async reloadMicroApp(name: string): Promise<boolean> {
    try {
      const microApp = this.microApps.get(name);
      if (microApp) {
        console.log('[QiankunService] Reloading micro app:', name);
        await microApp.unmount();
        await microApp.mount();
        console.log('[QiankunService] Micro app reloaded successfully:', name);
        return true;
      }
      return false;
    } catch (error) {
      console.error('[QiankunService] Failed to reload micro app:', name, error);
      return false;
    }
  }

  /**
   * 清理所有微前端应用
   */
  async cleanup(): Promise<void> {
    try {
      console.log('[QiankunService] Cleaning up all micro apps');
      
      for (const [name, microApp] of this.microApps) {
        try {
          await microApp.unmount();
          console.log('[QiankunService] Unmounted micro app:', name);
        } catch (error) {
          console.error('[QiankunService] Failed to unmount micro app:', name, error);
        }
      }
      
      this.microApps.clear();
      this.globalState = null;
      this.isStarted = false;
      this.qiankunInitialized = false;
      
      console.log('[QiankunService] Cleanup completed');
    } catch (error) {
      console.error('[QiankunService] Cleanup failed:', error);
    }
  }
}