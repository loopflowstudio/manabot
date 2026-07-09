import { defineConfig } from '@playwright/test';

// Ports are chosen to stay clear of the default dev setup (8000/5173) so the
// e2e suite can run alongside a live instance.
const apiPort = process.env.MANABOT_API_PORT ?? '8011';
const frontendPort = process.env.MANABOT_FRONTEND_PORT ?? '5183';
// The backend needs the repo venv's uvicorn (see AGENTS.md); override with
// MANABOT_UVICORN when testing against a different venv.
const uvicorn = process.env.MANABOT_UVICORN ?? '.venv/bin/uvicorn';

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://localhost:${frontendPort}`,
  },
  webServer: [
    {
      command: `${uvicorn} gui.server:app --port ${apiPort}`,
      cwd: '..',
      url: `http://localhost:${apiPort}/api/traces`,
      reuseExistingServer: true,
      timeout: 180_000,
    },
    {
      command: `npm run dev -- --port ${frontendPort} --strictPort`,
      env: { MANABOT_API_PORT: apiPort },
      url: `http://localhost:${frontendPort}`,
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
