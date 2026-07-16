import { defineConfig } from '@playwright/test';

// Ports are chosen to stay clear of the default dev setup (8000/5173) so the
// e2e suite can run alongside a live instance.
const apiPort = process.env.ETUDE_API_PORT ?? '8011';
const frontendPort = process.env.ETUDE_FRONTEND_PORT ?? '5183';
const externalServers = process.env.ETUDE_EXTERNAL_SERVERS === '1';
// The backend needs the repo venv's uvicorn (see AGENTS.md); override with
// ETUDE_UVICORN when testing against a different venv.
const uvicorn = process.env.ETUDE_UVICORN ?? '.venv/bin/uvicorn';

export default defineConfig({
  testDir: './e2e',
  testIgnore: ['release-prompt-matrix.spec.ts'],
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://localhost:${frontendPort}`,
  },
  webServer: externalServers
    ? undefined
    : [
        {
          command: `${uvicorn} gui.server:app --port ${apiPort}`,
          cwd: '..',
          url: `http://localhost:${apiPort}/api/traces`,
          reuseExistingServer: true,
          timeout: 180_000,
        },
        {
          command: `npm run dev -- --port ${frontendPort} --strictPort`,
          env: { ETUDE_API_PORT: apiPort },
          url: `http://localhost:${frontendPort}`,
          reuseExistingServer: true,
          timeout: 60_000,
        },
      ],
});
