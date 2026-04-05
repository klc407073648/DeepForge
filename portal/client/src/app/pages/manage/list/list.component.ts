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
import { TranslateModule } from '@ngx-translate/core';
import { AuthHttpService } from '../../../services/auth-http.service';
import { User } from '../../../model/user.model';

@Component({
  selector: 'app-Directives',
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
    TranslateModule,
  ],
  templateUrl: './list.component.html',
  styleUrl: './list.component.css',
})
export class ListComponent implements OnInit {
  userList: User[] = [];
  loading = false;
  total = 0;
  pageIndex = 1;
  pageSize = 10;
  searchUsername = '';
  isAdmin = false;

  constructor(
    private authService: AuthHttpService,
    private message: NzMessageService
  ) {}

  ngOnInit(): void {
    this.isAdmin = this.authService.isAdmin();
    this.loadUserList();
  }

  loadUserList(): void {
    this.loading = true;
    console.log('[DirectivesComponent] 开始加载用户列表，参数:', {
      page: this.pageIndex,
      pageSize: this.pageSize,
      username: this.searchUsername,
      isAdmin: this.isAdmin,
    });

    this.authService
      .getUserList({
        page: this.pageIndex,
        pageSize: this.pageSize,
        username: this.searchUsername,
      })
      .subscribe({
        next: (response) => {
          this.loading = false;
          console.log('[DirectivesComponent] 用户列表响应:', response);
          if (response.success && response.data) {
            this.userList = response.data.users;
            this.total = response.data.total;
            console.log('[DirectivesComponent] 成功加载用户列表，总数:', this.total);
          } else {
            console.error('[DirectivesComponent] 获取用户列表失败:', response.message);
            this.message.error(response.message || '获取用户列表失败');
          }
        },
        error: (error) => {
          this.loading = false;
          console.error('[DirectivesComponent] 请求出错:', error);
          this.message.error(error.message || '获取用户列表失败');
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
}
