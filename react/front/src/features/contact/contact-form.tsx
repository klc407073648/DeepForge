"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const contactSchema = z.object({
  name: z.string().min(2, "请输入至少 2 个字符"),
  email: z.string().email("请输入有效邮箱")
});

type ContactValues = z.infer<typeof contactSchema>;

export function ContactForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset
  } = useForm<ContactValues>({
    resolver: zodResolver(contactSchema),
    defaultValues: {
      name: "",
      email: ""
    }
  });

  const onSubmit = async (values: ContactValues) => {
    console.log("contact", values);
    reset();
  };

  return (
    <form className="form-grid" onSubmit={handleSubmit(onSubmit)}>
      <div className="field-group">
        <label className="field-label" htmlFor="name">
          姓名
        </label>
        <Input id="name" placeholder="张三" {...register("name")} />
        {errors.name ? <p className="field-error">{errors.name.message}</p> : null}
      </div>
      <div className="field-group">
        <label className="field-label" htmlFor="email">
          邮箱
        </label>
        <Input id="email" placeholder="name@example.com" {...register("email")} />
        {errors.email ? <p className="field-error">{errors.email.message}</p> : null}
      </div>
      <Button type="submit" disabled={isSubmitting}>
        提交
      </Button>
    </form>
  );
}
