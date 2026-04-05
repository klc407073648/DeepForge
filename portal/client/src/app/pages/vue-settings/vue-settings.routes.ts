import { Routes } from '@angular/router';
import { VueSettingsComponent } from './vue-settings.component';
import { AuthGuard } from '../../guards/auth.guard';

export const VUE_SETTINGS_ROUTES: Routes = [
  {
    path: '',
    component: VueSettingsComponent,
    canActivate: [AuthGuard]
  }
];

