import { defineConfig } from '@playwright/test';

// Ports are chosen to stay clear of the default dev setup (8000/5173) so the
// e2e suite can run alongside a live instance.
const apiPort = process.env.ACADEMY_API_PORT ?? '8011';
const frontendPort = process.env.ACADEMY_FRONTEND_PORT ?? '5183';
const externalServers = process.env.ACADEMY_EXTERNAL_SERVERS === '1';
// The backend must run through the repository's uv environment (see AGENTS.md).
// ACADEMY_UVICORN can still replace the complete command for another environment.
const uvicorn = process.env.ACADEMY_UVICORN ?? 'uv run --active --no-sync uvicorn';

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
          command: `${uvicorn} academy.server:app --port ${apiPort}`,
          cwd: '..',
          url: `http://localhost:${apiPort}/api/traces`,
          reuseExistingServer: true,
          timeout: 180_000,
        },
        {
          command: `npm run dev -- --port ${frontendPort} --strictPort`,
          env: { ACADEMY_API_PORT: apiPort },
          url: `http://localhost:${frontendPort}`,
          reuseExistingServer: true,
          timeout: 60_000,
        },
      ],
});
