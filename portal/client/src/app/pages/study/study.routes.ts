import { Routes } from '@angular/router';
import { StudyComponent } from './study.component';
import { DirectivesComponent } from './Directives/directives.component';
import { AuthGuard } from '../../guards/auth.guard';
import { AdminGuard } from '../../guards/admin.guard';

export const STUDY_ROUTES: Routes = [
  {
    path: '',
    component: StudyComponent,
    data: { module: 'study' },
    canActivate: [AuthGuard, AdminGuard],
  },
  {
    path: 'Directives',
    component: DirectivesComponent,
    data: { module: 'study' },
    canActivate: [AuthGuard],
  },
];
