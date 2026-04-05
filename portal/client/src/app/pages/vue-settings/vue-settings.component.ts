import {
  Component,
  OnInit,
  OnDestroy,
  ElementRef,
  ViewChild,
  AfterViewInit,
  Inject,
  PLATFORM_ID,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { isPlatformBrowser } from '@angular/common';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { TranslateModule } from '@ngx-translate/core';
import { QiankunService } from '../../services/qiankun.service';

@Component({
  selector: 'app-vue-settings',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    NzButtonModule,
    NzIconModule,
    TranslateModule,
  ],
  templateUrl: './vue-settings.component.html',
  styleUrls: ['./vue-settings.component.css'],
})
export class VueSettingsComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('vueContainer', { static: true }) vueContainer!: ElementRef;
  isLoading = true;
  loadError = '';

  constructor(
    private qiankunService: QiankunService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngOnInit() {
    console.log('VueSettingsComponent ngOnInit called');

    // 只在浏览器环境中监听消息
    if (isPlatformBrowser(this.platformId)) {
      window.addEventListener('message', this.handleMessage.bind(this));
      
      // 预检查Vue应用是否可用
      this.checkVueAppAvailability();
    }
  }

  ngAfterViewInit() {
    console.log('VueSettingsComponent ngAfterViewInit called');

    // 延迟加载Vue应用，确保容器已渲染
    setTimeout(() => {
      console.log('Starting to load Vue app...');
      this.loadVueApp();
    }, 100);
  }

  ngOnDestroy() {
    // 清理Vue应用
    this.qiankunService.unmountVueSettingsApp();
    if (isPlatformBrowser(this.platformId)) {
      window.removeEventListener('message', this.handleMessage.bind(this));
    }
  }


  async checkVueAppAvailability() {
    try {
      console.log('Checking Vue app availability at http://localhost:8081');
      const response = await fetch('http://localhost:8081', { 
        method: 'HEAD',
        mode: 'no-cors'
      });
      console.log('Vue app is available');
      return true;
    } catch (error) {
      console.warn('Vue app not available at http://localhost:8081:', error);
      return false;
    }
  }

  async loadVueApp() {
    try {
      console.log('Loading Vue app into container: #vue-settings-container');
      this.isLoading = true;
      this.loadError = '';

      // 检查容器元素是否存在
      const container = document.querySelector('#vue-settings-container');
      console.log('Container element found:', !!container);

      if (!container) {
        console.error('Container element #vue-settings-container not found');
        this.loadError = '容器元素未找到';
        this.isLoading = false;
        return;
      }

      // 检查Vue应用是否可用
      const isAvailable = await this.checkVueAppAvailability();
      if (!isAvailable) {
        this.loadError = 'Vue应用未启动，请先启动Vue应用：\n1. 打开命令行\n2. 运行: cd vue-settings-app && npm run serve\n3. 等待Vue应用在8081端口启动';
        this.isLoading = false;
        return;
      }

      const result = await this.qiankunService.loadVueSettingsApp(
        '#vue-settings-container'
      );
      console.log('Vue app loaded successfully:', result);

      if (!result) {
        console.error('Vue应用加载失败，请检查Vue应用是否在8081端口运行');
        this.loadError = 'Vue应用加载失败，请检查控制台错误信息';
      }
      
      this.isLoading = false;
    } catch (error) {
      console.error('加载Vue应用失败:', error);
      this.loadError = error instanceof Error ? error.message : String(error);
      this.isLoading = false;
    }
  }

  private handleMessage(event: MessageEvent) {
    console.log('Received message:', event.data);

    if (event.data && event.data.type === 'NAVIGATE_TO_MAIN') {
      this.goBack();
    } else if (event.data && event.data.type === 'ROUTE_CHANGE') {
      // 处理Vue应用的路由变化
      console.log('Vue route changed to:', event.data.path);
      // 更新Angular路由
      if (isPlatformBrowser(this.platformId)) {
        // 使用replaceState避免在浏览器历史中创建新条目
        window.history.replaceState(null, '', event.data.path);
      }
    }
  }

  openVueAppDirectly() {
    if (isPlatformBrowser(this.platformId)) {
      window.open('http://localhost:8081', '_blank');
    }
  }

  goBack() {
    // 返回主应用 - 现在使用 routerLink，这个方法可能不会被调用
    if (isPlatformBrowser(this.platformId)) {
      window.history.back();
    }
  }
}
