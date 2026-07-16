import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const frontendDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const apiPort = process.env.MANABOT_RELEASE_API_PORT ?? '8012';
const frontendPort = process.env.MANABOT_RELEASE_FRONTEND_PORT ?? '5184';
const outputDir = fileURLToPath(new URL('./test-results/release', import.meta.url));
const traceDir = fileURLToPath(new URL('./test-results/release/traces', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  testMatch: 'release-prompt-matrix.spec.ts',
  outputDir,
  timeout: 600_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
  },
  webServer: [
    {
      command: `uv run uvicorn gui.server:app --host 127.0.0.1 --port ${apiPort}`,
      cwd: repoRoot,
      env: { MANABOT_GUI_TRACES_DIR: traceDir },
      url: `http://127.0.0.1:${apiPort}/api/traces`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npm run preview -- --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      cwd: frontendDir,
      env: { MANABOT_API_PORT: apiPort },
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
