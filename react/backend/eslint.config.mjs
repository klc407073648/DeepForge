import { FlatCompat } from "@eslint/eslintrc";
import { dirname } from "path";
import { fileURLToPath } from "url";

const filename = fileURLToPath(import.meta.url);
const dirnamePath = dirname(filename);

const compat = new FlatCompat({
  baseDirectory: dirnamePath
});

export default [...compat.extends("next/core-web-vitals", "next/typescript")];
