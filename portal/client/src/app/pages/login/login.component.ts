import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { Router } from '@angular/router';
import { NzFormModule } from 'ng-zorro-antd/form';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzInputGroupComponent } from 'ng-zorro-antd/input';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzMessageModule, NzMessageService } from 'ng-zorro-antd/message';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { AuthHttpService } from '../../services/auth-http.service';
import { LoginRequest, UserRole } from '../../model/user.model';
import { passwordValidator, usernameValidator } from '../../common/validator';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { ValidationMessageComponent } from '../../components/validation-message/validation-message.component';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    NzFormModule,
    NzInputModule,
    NzInputGroupComponent,
    NzButtonModule,
    NzCardModule,
    NzMessageModule,
    NzIconModule,
    TranslateModule,
    ValidationMessageComponent,
  ],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent implements OnInit {
  loginForm!: FormGroup;
  loading = false;
  passwordVisible = false;

  constructor(
    private authService: AuthHttpService,
    private message: NzMessageService,
    private router: Router,
    private fb: FormBuilder,
    private translate: TranslateService
  ) {
    this.loginForm = this.fb.group({
      username: [null, [Validators.required, usernameValidator]],
      password: [null, [Validators.required, passwordValidator]],
      remember: [true],
    });
  }

  ngOnInit(): void {
    // 检查用户是否已登录，如果已登录则跳转到相应页面
    if (this.authService.isLoggedIn()) {
      const currentUser = this.authService.getCurrentUser();
      if (currentUser) {
        this.router.navigate(['/welcome']);
      }
    }
  }

  onSubmit(): void {
    if (
      !this.loginForm.get('username')?.value ||
      !this.loginForm.get('password')?.value
    ) {
      this.message.error(this.translate.instant('login.fillCompleteInfo'));
      return;
    }

    this.loading = true;
    const loginRequest: LoginRequest = {
      username: this.loginForm.get('username')?.value,
      password: this.loginForm.get('password')?.value,
    };

    this.authService.login(loginRequest).subscribe({
      next: (result) => {
        this.loading = false;
        if (result.success && result.data) {
          this.message.success(this.translate.instant('login.loginSuccess'));
          // 根据用户角色重定向到不同页面
          this.router.navigate(['/welcome']);
        } else {
          this.message.error(
            result.message || this.translate.instant('login.invalidCredentials')
          );
        }
      },
      error: (error) => {
        this.loading = false;
        this.message.error(
          error.message || this.translate.instant('login.loginFailed')
        );
      },
    });
  }

  goToRegister(): void {
    this.router.navigate(['/register']);
  }
}
