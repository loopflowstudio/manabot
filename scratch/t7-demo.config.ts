import { defineConfig } from '../academy/ui/node_modules/@playwright/test';

export default defineConfig({
  testDir: '.', testMatch: 't7-demo.spec.ts', workers: 1,
  timeout: 180_000, outputDir: '../.cache/mtg226-demo/browser',
  use: { baseURL: 'http://127.0.0.1:5296', trace: 'retain-on-failure' },
  webServer: [
    {
      command: 'uv run --no-sync uvicorn academy.server:app --host 127.0.0.1 --port 8126',
      cwd: '..', url: 'http://127.0.0.1:8126/api/traces',
      env: { ACADEMY_TRACES_DIR: '.cache/mtg226-demo/traces' },
      reuseExistingServer: false, timeout: 60_000,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5296 --strictPort',
      cwd: '../academy/ui', url: 'http://127.0.0.1:5296',
      env: { ACADEMY_API_PORT: '8126' },
      reuseExistingServer: false, timeout: 60_000,
    },
  ],
});
