import { UserRole } from '../model/user.model';

// menu.config.ts
export interface MenuItem {
  title: string;
  link: string;
  icon: string;
  permission?: string; // 添加权限字段
  roles?: (UserRole.Admin | UserRole.User)[]; // 添加角色字段
}

export const MENU_CONFIG: { [key: string]: MenuItem[] } = {
  manage: [
    {
      title: 'menu.userList',
      link: '/manage/list',
      icon: UserRole.User,
      permission: 'manage.list.view',
      roles: [UserRole.Admin, UserRole.User],
    },
    {
      title: 'menu.roleManagement',
      link: '/manage/roles',
      icon: 'team',
      permission: 'manage.roles.view',
      roles: [UserRole.Admin],
    },
    {
      title: 'menu.permissionSettings',
      link: '/manage/permissions',
      icon: 'safety-certificate',
      permission: 'manage.permissions.view',
      roles: [UserRole.Admin],
    },
  ],
  study: [
    {
      title: 'menu.directives',
      link: '/study/Directives',
      icon: 'code',
      permission: 'manage.Directives.view',
      roles: [UserRole.Admin, UserRole.User],
    },
  ],
  settings: [
    {
      title: 'menu.basicSettings',
      link: '/settings/basic',
      icon: 'setting',
      permission: 'settings.basic.view',
      roles: [UserRole.Admin],
    },
    {
      title: 'menu.emailConfig',
      link: '/settings/email',
      icon: 'mail',
      permission: 'settings.email.view',
      roles: [UserRole.Admin],
    },
    {
      title: 'menu.logView',
      link: '/settings/logs',
      icon: 'file-search',
      permission: 'settings.logs.view',
      roles: [UserRole.Admin],
    },
  ],
};
