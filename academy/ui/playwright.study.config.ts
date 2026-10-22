import { defineConfig } from '@playwright/test';

const apiPort = process.env.ACADEMY_API_PORT ?? '8012';
const frontendPort = process.env.ACADEMY_FRONTEND_PORT ?? '5184';

export default defineConfig({
  testDir: './e2e',
  testMatch: 'study.spec.ts',
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://localhost:${frontendPort}`,
  },
  webServer: [
    {
      command: `uv run --active --no-sync uvicorn study_server:app --app-dir academy/ui/e2e --port ${apiPort}`,
      cwd: '../..',
      url: `http://localhost:${apiPort}/api/traces`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npm run dev -- --port ${frontendPort} --strictPort`,
      env: { ACADEMY_API_PORT: apiPort },
      url: `http://localhost:${frontendPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
