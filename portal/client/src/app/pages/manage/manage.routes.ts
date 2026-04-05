import { Routes } from '@angular/router';
import { ManageComponent } from './manage.component';
import { ListComponent } from './list/list.component';
import { PermissionComponent } from './permission/permission.component';
import { RoleComponent } from './roles/role.component';
import { AuthGuard } from '../../guards/auth.guard';
import { AdminGuard } from '../../guards/admin.guard';

export const MANAGE_ROUTES: Routes = [
  {
    path: '',
    component: ManageComponent,
    data: { module: 'manage' },
    canActivate: [AuthGuard, AdminGuard],
  },
  {
    path: 'list',
    component: ListComponent,
    data: { module: 'manage' },
    canActivate: [AuthGuard],
  },
  {
    path: 'roles',
    component: RoleComponent,
    data: { module: 'manage' },
    canActivate: [AuthGuard, AdminGuard],
  },
  {
    path: 'permissions',
    component: PermissionComponent,
    data: { module: 'manage' },
    canActivate: [AuthGuard, AdminGuard],
  },
];
