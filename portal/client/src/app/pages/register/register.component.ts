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
import { RegisterRequest, UserRole } from '../../model/user.model';
import {
  passwordValidator,
  usernameValidator,
  confirmPasswordValidator,
} from '../../common/validator';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { ValidationMessageComponent } from '../../components/validation-message/validation-message.component';

@Component({
  selector: 'app-register',
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
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css'],
})
export class RegisterComponent implements OnInit {
  registerForm!: FormGroup;
  loading = false;
  passwordVisible = false;
  confirmPasswordVisible = false;

  constructor(
    private authService: AuthHttpService,
    private message: NzMessageService,
    private router: Router,
    private fb: FormBuilder,
    private translate: TranslateService
  ) {
    this.registerForm = this.fb.group({
      username: [null, [Validators.required, usernameValidator]],
      password: [null, [Validators.required, passwordValidator]],
      confirmPassword: [
        null,
        [Validators.required, confirmPasswordValidator('password')],
      ],
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
    // 标记所有字段为已触碰，以显示验证错误
    Object.values(this.registerForm.controls).forEach((control) => {
      control.markAsTouched();
      control.updateValueAndValidity();
    });

    if (this.registerForm.invalid) {
      this.message.error(this.translate.instant('register.fillCompleteInfo'));
      return;
    }

    this.loading = true;
    const registerRequest: RegisterRequest = {
      username: this.registerForm.get('username')?.value,
      password: this.registerForm.get('password')?.value,
    };

    this.authService.register(registerRequest).subscribe({
      next: (result) => {
        this.loading = false;
        if (result.success && result.data) {
          this.message.success(
            this.translate.instant('register.registerSuccess')
          );
          // 根据用户角色重定向到不同页面
          if (result.data.user.role === UserRole.Admin) {
            this.router.navigate(['/manage']);
          } else {
            this.router.navigate(['/welcome']);
          }
        } else {
          this.message.error(
            result.message || this.translate.instant('register.registerFailed')
          );
        }
      },
      error: (error) => {
        this.loading = false;
        this.message.error(
          error.message || this.translate.instant('register.registerFailed')
        );
      },
    });
  }

  goToLogin(): void {
    this.router.navigate(['/login']);
  }
}
