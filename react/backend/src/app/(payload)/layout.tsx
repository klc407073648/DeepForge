import type { ReactNode } from "react";
import "@payloadcms/next/css";

export default function PayloadLayout({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
