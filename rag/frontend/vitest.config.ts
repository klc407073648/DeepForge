import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(
  viteConfig,
  defineConfig({
    esbuild: {
      jsx: 'automatic',
    },
    test: {
      environment: 'jsdom',
      setupFiles: ['./test/setup.ts'],
      globals: false,
      css: true,
      include: ['src/**/*.{test,spec}.{ts,tsx}', 'test/**/*.{test,spec}.{ts,tsx}'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'html', 'json-summary'],
        reportsDirectory: './coverage',
        include: ['src/**/*.{ts,tsx}'],
        exclude: [
          'src/**/*.test.{ts,tsx}',
          'src/**/*.spec.{ts,tsx}',
          'src/main.tsx',
          'src/vite-env.d.ts',
          'src/types/**',
        ],
      },
    },
  }),
)
