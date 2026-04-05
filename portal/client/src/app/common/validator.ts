import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';

// 密码必须包含：大写字母、小写字母、数字、特殊字符，且长度 >= 8
export function passwordValidator(
  control: AbstractControl
): ValidationErrors | null {
  const value = control.value;
  if (!value) return null; // 允许空值由 required 处理

  const hasUpperCase = /[A-Z]/.test(value);
  const hasLowerCase = /[a-z]/.test(value);
  const hasNumber = /\d/.test(value);
  const isValidLength = value.length >= 8;

  const passwordValid =
    hasUpperCase && hasLowerCase && hasNumber && isValidLength;

  return !passwordValid
    ? {
        passwordStrength: {
          message: 'validation.passwordStrength',
          hasUpperCase,
          hasLowerCase,
          hasNumber,
          isValidLength,
        },
      }
    : null;
}

// 密码确认验证器
export function confirmPasswordValidator(passwordField: string): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    if (!control.parent) return null;

    const password = control.parent.get(passwordField);
    const confirmPassword = control;

    if (!password || !confirmPassword) return null;
    if (!confirmPassword.value) return null;

    if (password.value !== confirmPassword.value) {
      return {
        passwordMismatch: {
          message: 'validation.passwordMismatch',
        },
      };
    }

    return null;
  };
}

// 用户名只能包含字母、数字、下划线，长度 3-20
export function usernameValidator(
  control: AbstractControl
): ValidationErrors | null {
  const value = control.value;
  if (!value) return null;

  const isValidFormat = /^[a-zA-Z0-9_]+$/.test(value);
  const isValidLength = value.length >= 3 && value.length <= 20;

  if (!isValidFormat) {
    return {
      invalidChars: {
        message: 'validation.usernameInvalidChars',
      },
    };
  }
  if (!isValidLength) {
    return {
      invalidLength: {
        message: 'validation.usernameInvalidLength',
      },
    };
  }

  return null;
}
