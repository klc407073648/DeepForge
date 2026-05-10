"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuthStore } from "@/stores/auth-store";

const registerSchema = z
  .object({
    name: z.string().min(2, "请输入至少 2 个字符"),
    email: z.string().email("请输入有效邮箱"),
    password: z.string().min(6, "密码至少 6 位"),
    confirmPassword: z.string().min(6, "请再次输入密码")
  })
  .refine((values) => values.password === values.confirmPassword, {
    message: "两次输入的密码不一致",
    path: ["confirmPassword"]
  });

type RegisterValues = z.infer<typeof registerSchema>;

export default function RegisterPage() {
  const router = useRouter();
  const registerUser = useAuthStore((state) => state.register);
  const [message, setMessage] = useState("");

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting }
  } = useForm<RegisterValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      name: "",
      email: "",
      password: "",
      confirmPassword: ""
    }
  });

  const onSubmit = (values: RegisterValues) => {
    const result = registerUser({
      name: values.name,
      email: values.email,
      password: values.password
    });

    if (!result.ok) {
      setMessage(result.message ?? "注册失败，请稍后重试。");
      return;
    }

    router.replace("/portal");
  };

  return (
    <main className="auth-page">
      <Card className="auth-card">
        <CardHeader>
          <CardDescription>DeepForge Portal</CardDescription>
          <CardTitle className="text-3xl">创建账号</CardTitle>
          <CardDescription>注册后会直接进入内容管理门户。</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="form-grid" onSubmit={handleSubmit(onSubmit)}>
            <div className="field-group">
              <label className="field-label" htmlFor="name">
                姓名
              </label>
              <Input id="name" {...register("name")} />
              {errors.name ? <p className="field-error">{errors.name.message}</p> : null}
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="email">
                邮箱
              </label>
              <Input id="email" type="email" {...register("email")} />
              {errors.email ? <p className="field-error">{errors.email.message}</p> : null}
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="password">
                密码
              </label>
              <Input id="password" type="password" {...register("password")} />
              {errors.password ? <p className="field-error">{errors.password.message}</p> : null}
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="confirmPassword">
                确认密码
              </label>
              <Input id="confirmPassword" type="password" {...register("confirmPassword")} />
              {errors.confirmPassword ? <p className="field-error">{errors.confirmPassword.message}</p> : null}
            </div>
            {message ? <p className="field-error">{message}</p> : null}
            <Button type="submit" disabled={isSubmitting}>
              注册并进入门户
            </Button>
          </form>
          <p className="auth-footer">
            已有账号？
            <Link href="/login">返回登录</Link>
          </p>
        </CardContent>
      </Card>
    </main>
  );
}
