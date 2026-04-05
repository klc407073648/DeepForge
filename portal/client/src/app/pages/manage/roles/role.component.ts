import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NzTableModule } from 'ng-zorro-antd/table';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzMessageService } from 'ng-zorro-antd/message';
import { NzTagModule } from 'ng-zorro-antd/tag';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzSelectModule } from 'ng-zorro-antd/select';
import { NzPopconfirmModule } from 'ng-zorro-antd/popconfirm';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { AuthHttpService } from '../../../services/auth-http.service';
import { User, UserRole } from '../../../model/user.model';

@Component({
  selector: 'app-role',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    NzTableModule,
    NzButtonModule,
    NzInputModule,
    NzTagModule,
    NzCardModule,
    NzIconModule,
    NzSelectModule,
    NzPopconfirmModule,
    TranslateModule,
  ],
  templateUrl: './role.component.html',
  styleUrl: './role.component.css',
})
export class RoleComponent implements OnInit {
  userList: User[] = [];
  loading = false;
  total = 0;
  pageIndex = 1;
  pageSize = 10;
  searchUsername = '';
  currentUser: User | null = null;

  // 角色选项
  roleOptions = [
    { label: 'common.admin', value: UserRole.Admin },
    { label: 'common.user', value: UserRole.User },
  ];

  constructor(
    private authService: AuthHttpService,
    private message: NzMessageService,
    private translate: TranslateService
  ) {}

  ngOnInit(): void {
    this.currentUser = this.authService.getCurrentUser();
    this.loadUserList();
  }

  loadUserList(): void {
    this.loading = true;
    this.authService
      .getUserList({
        page: this.pageIndex,
        pageSize: this.pageSize,
        username: this.searchUsername,
      })
      .subscribe({
        next: (response) => {
          this.loading = false;
          if (response.success && response.data) {
            this.userList = response.data.users;
            this.total = response.data.total;
          } else {
            this.message.error(response.message || this.translate.instant('roleManagement.loadFailed'));
          }
        },
        error: (error) => {
          this.loading = false;
          this.message.error(error.message || this.translate.instant('roleManagement.loadFailed'));
        },
      });
  }

  onPageChange(pageIndex: number): void {
    this.pageIndex = pageIndex;
    this.loadUserList();
  }

  onPageSizeChange(pageSize: number): void {
    this.pageSize = pageSize;
    this.pageIndex = 1;
    this.loadUserList();
  }

  onSearch(): void {
    this.pageIndex = 1;
    this.loadUserList();
  }

  onReset(): void {
    this.searchUsername = '';
    this.pageIndex = 1;
    this.loadUserList();
  }

  /**
   * 判断用户是否可以修改角色
   */
  canEditRole(user: User): boolean {
    // admin 账号不能修改
    if (user.username === 'admin') {
      return false;
    }
    // 不能修改自己
    if (this.currentUser && user.id === this.currentUser.id) {
      return false;
    }
    return true;
  }

  /**
   * 更新用户角色
   */
  updateRole(user: User, newRole: UserRole): void {
    const oldRole = user.role;
    
    this.authService
      .updateUserRole({
        userId: user.id,
        role: newRole,
      })
      .subscribe({
        next: (response) => {
          if (response.success) {
            this.message.success(this.translate.instant('roleManagement.updateSuccess'));
            // 更新本地数据
            user.role = newRole;
          } else {
            this.message.error(response.message || this.translate.instant('roleManagement.updateFailed'));
            // 恢复原角色
            user.role = oldRole;
          }
        },
        error: (error) => {
          this.message.error(error.message || this.translate.instant('roleManagement.updateFailed'));
          // 恢复原角色
          user.role = oldRole;
        },
      });
  }

  /**
   * 获取角色标签颜色
   */
  getRoleColor(role: string): string {
    return role === UserRole.Admin ? 'red' : 'blue';
  }

  /**
   * 获取角色显示文本
   */
  getRoleText(role: string): string {
    return role === UserRole.Admin
      ? this.translate.instant('common.admin')
      : this.translate.instant('common.user');
  }
}
