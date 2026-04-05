import { Routes } from '@angular/router';
import { AuthGuard } from './guards/auth.guard';
import { AdminGuard } from './guards/admin.guard';

export const routes: Routes = [
  {
    path: 'login',
    loadChildren: () =>
      import('./pages/login/login.routes').then((m) => m.LOGIN_ROUTES),
  },
  {
    path: 'register',
    loadChildren: () =>
      import('./pages/register/register.routes').then((m) => m.REGISTER_ROUTES),
  },
  {
    path: '',
    pathMatch: 'full',
    redirectTo: '/login',
  },
  {
    path: 'welcome',
    loadChildren: () =>
      import('./pages/welcome/welcome.routes').then((m) => m.WELCOME_ROUTES),
    canActivate: [AuthGuard],
  },
  {
    path: 'manage',
    loadChildren: () =>
      import('./pages/manage/manage.routes').then((m) => m.MANAGE_ROUTES),
    canActivate: [AuthGuard],
  },
  {
    path: 'study',
    loadChildren: () =>
      import('./pages/study/study.routes').then((m) => m.STUDY_ROUTES),
    canActivate: [AuthGuard],
  },
  {
    path: 'vue-settings',
    loadChildren: () =>
      import('./pages/vue-settings/vue-settings.routes').then((m) => m.VUE_SETTINGS_ROUTES),
    canActivate: [AuthGuard],
  }
];
