import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const frontendDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = fileURLToPath(new URL('../..', import.meta.url));
const apiPort = process.env.ACADEMY_RELEASE_API_PORT ?? '8012';
const frontendPort = process.env.ACADEMY_RELEASE_FRONTEND_PORT ?? '5184';
const outputDir = fileURLToPath(new URL('./test-results/release', import.meta.url));
const traceDir = fileURLToPath(new URL('./test-results/release/traces', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  testMatch: ['decision-surface.spec.ts', 'learn.spec.ts', 'combat-presentation.spec.ts', 'choice-navigation.spec.ts', 'elemental-world.spec.ts', 'deck-browser.spec.ts', 'decklist-import.spec.ts', 'trigger-order.spec.ts'],
  outputDir,
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    browserName: 'chromium',
    colorScheme: 'dark',
    deviceScaleFactor: 1,
    locale: 'en-US',
    reducedMotion: 'reduce',
    timezoneId: 'UTC',
    viewport: { width: 1600, height: 1200 },
  },
  webServer: [
    {
      command: `uv run --active --no-sync python -m uvicorn trigger_order_server:app --app-dir tests/fixtures --host 127.0.0.1 --port ${apiPort}`,
      cwd: repoRoot,
      env: { ACADEMY_TRACES_DIR: traceDir },
      url: `http://127.0.0.1:${apiPort}/api/traces`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npm run preview -- --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      cwd: frontendDir,
      env: { ACADEMY_API_PORT: apiPort },
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
