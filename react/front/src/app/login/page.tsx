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

const loginSchema = z.object({
  email: z.string().email("请输入有效邮箱"),
  password: z.string().min(6, "密码至少 6 位")
});

type LoginValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter();
  const login = useAuthStore((state) => state.login);
  const [message, setMessage] = useState("");

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting }
  } = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "admin@deepforge.dev",
      password: "123456"
    }
  });

  const onSubmit = (values: LoginValues) => {
    const ok = login(values.email, values.password);

    if (!ok) {
      setMessage("邮箱或密码不正确。");
      return;
    }

    router.replace("/portal");
  };

  return (
    <main className="auth-page">
      <Card className="auth-card">
        <CardHeader>
          <CardDescription>DeepForge Portal</CardDescription>
          <CardTitle className="text-3xl">登录门户</CardTitle>
          <CardDescription>使用演示账号或注册新账号进入内容管理门户。</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="form-grid" onSubmit={handleSubmit(onSubmit)}>
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
            {message ? <p className="field-error">{message}</p> : null}
            <Button type="submit" disabled={isSubmitting}>
              登录
            </Button>
          </form>
          <p className="auth-footer">
            还没有账号？
            <Link href="/register">立即注册</Link>
          </p>
        </CardContent>
      </Card>
    </main>
  );
}
