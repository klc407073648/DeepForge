import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ContactForm } from "./contact-form";

describe("ContactForm", () => {
  it("shows validation messages", async () => {
    const user = userEvent.setup();
    render(<ContactForm />);

    await user.click(screen.getByRole("button", { name: "提交" }));

    expect(await screen.findByText("请输入至少 2 个字符")).toBeInTheDocument();
    expect(await screen.findByText("请输入有效邮箱")).toBeInTheDocument();
  });
});
