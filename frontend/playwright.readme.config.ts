import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const frontendDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const apiPort = process.env.ETUDE_README_API_PORT ?? '8013';
const frontendPort = process.env.ETUDE_README_FRONTEND_PORT ?? '5185';
const outputDir = fileURLToPath(new URL('./test-results/readme', import.meta.url));
const traceDir = fileURLToPath(new URL('./test-results/readme/traces', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  testMatch: 'readme-capture.spec.ts',
  outputDir,
  snapshotPathTemplate: `${repoRoot}/docs/assets/{arg}{ext}`,
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  expect: {
    toHaveScreenshot: {
      animations: 'disabled',
      caret: 'hide',
      maxDiffPixels: 0,
      scale: 'css',
      threshold: 0.2,
    },
  },
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    browserName: 'chromium',
    colorScheme: 'dark',
    deviceScaleFactor: 1,
    locale: 'en-US',
    reducedMotion: 'reduce',
    timezoneId: 'UTC',
    viewport: { width: 760, height: 1000 },
  },
  webServer: [
    {
      command: `uv run --active --no-sync uvicorn etude.server:app --host 127.0.0.1 --port ${apiPort}`,
      cwd: repoRoot,
      env: { ETUDE_TRACES_DIR: traceDir },
      url: `http://127.0.0.1:${apiPort}/api/traces`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      cwd: frontendDir,
      env: { ETUDE_API_PORT: apiPort },
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
