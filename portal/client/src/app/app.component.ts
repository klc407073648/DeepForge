// app.component.ts
import { Component, CUSTOM_ELEMENTS_SCHEMA, OnInit } from '@angular/core';
import {
  Router,
  ActivatedRoute,
  NavigationEnd,
  RouterLink,
  RouterOutlet,
} from '@angular/router';
import { MENU_CONFIG, MenuItem } from './conf/menu.config';
import { filter } from 'rxjs/operators';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzLayoutModule } from 'ng-zorro-antd/layout';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { NzToolTipModule } from 'ng-zorro-antd/tooltip';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzDropDownModule } from 'ng-zorro-antd/dropdown';
import { NzAvatarModule } from 'ng-zorro-antd/avatar';
import { NgFor, NgIf } from '@angular/common';
import { AuthHttpService } from './services/auth-http.service';
import { User, UserRole } from './model/user.model';
import { LanguageSwitcherComponent } from './components/language-switcher/language-switcher.component';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { QiankunService } from './services/qiankun.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  imports: [
    RouterLink,
    RouterOutlet,
    NzButtonModule,
    NzIconModule,
    NzLayoutModule,
    NzMenuModule,
    NzToolTipModule,
    NzDropDownModule,
    NzAvatarModule,
    LanguageSwitcherComponent,
    TranslateModule,
    NgIf,
    NgFor,
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class AppComponent implements OnInit {
  currentModule: string | null = null;
  sidebarMenu: MenuItem[] = [];
  isCollapsed = false;
  currentUser: User | null = null;
  /** 登录/注册页全屏展示，不使用顶栏与内容区白底包裹 */
  isAuthShell = false;

  // 模块标题映射
  private moduleTitles: { [key: string]: string } = {
    manage: '用户管理',
    orders: '订单管理',
    settings: '系统设置',
  };

  toggleCollapsed(): void {
    this.isCollapsed = !this.isCollapsed;
  }

  constructor(
    private router: Router,
    private activatedRoute: ActivatedRoute,
    private authService: AuthHttpService,
    private translate: TranslateService,
    private qiankunService: QiankunService
  ) {
    console.log('AppComponent constructor called, qiankunService injected');
  }

  ngOnInit() {
    // 初始化翻译服务
    this.translate.use('zh');

    // 监听用户状态变化
    this.authService.currentUser$.subscribe((user) => {
      this.currentUser = user;
      console.log('[app.component] this.currentUser:', this.currentUser);
      if (this.currentUser) {
        console.log('[app.component] this.currentUser1111:', this.currentUser);
      }
    });

    // 监听路由变化
    this.router.events
      .pipe(filter((event) => event instanceof NavigationEnd))
      .subscribe(() => {
        this.updateSidebar();
      });

    // 初始化（首次加载）
    this.updateSidebar();
  }

  private updateAuthShellFlag(): void {
    const path = this.router.url.split('?')[0];
    this.isAuthShell =
      path === '/login' ||
      path === '/register' ||
      path.startsWith('/login/') ||
      path.startsWith('/register/');
  }

  private updateSidebar() {
    this.updateAuthShellFlag();

    let route = this.activatedRoute;
    while (route.firstChild) {
      route = route.firstChild;
    }

    // 获取当前路由的 data.module
    const data = route.snapshot.data;
    const module = data['module'];

    if (module && MENU_CONFIG[module]) {
      this.currentModule = module;
      // 根据用户权限过滤菜单
      this.sidebarMenu = this.filterMenuByPermission(MENU_CONFIG[module]);
    } else {
      this.currentModule = null;
      this.sidebarMenu = [];
    }
  }

  private filterMenuByPermission(menuItems: MenuItem[]): MenuItem[] {
    if (!this.currentUser) return [];

    return menuItems.filter((item) => {
      // 检查角色权限
      if (item.roles && !item.roles.includes(this.currentUser!.role)) {
        return false;
      }

      // 检查具体权限
      if (item.permission && !this.authService.hasPermission(item.permission)) {
        return false;
      }

      return true;
    });
  }

  getModuleTitle(): string {
    return this.moduleTitles[this.currentModule!] || '未知模块';
  }

  logout(): void {
    this.authService.logout().subscribe({
      next: () => {
        this.router.navigate(['/login']);
      },
      error: () => {
        // 即使登出失败，也跳转到登录页
        this.router.navigate(['/login']);
      },
    });
  }

  getUserDisplayName(): string {
    return this.currentUser?.username || '未知用户';
  }

  getUserRoleText(): string {
    return this.currentUser?.role === UserRole.Admin
      ? this.translate.instant('common.admin')
      : this.translate.instant('common.user');
  }
}
