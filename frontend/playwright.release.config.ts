import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const frontendDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const apiPort = process.env.ETUDE_RELEASE_API_PORT ?? '8012';
const frontendPort = process.env.ETUDE_RELEASE_FRONTEND_PORT ?? '5184';
const outputDir = fileURLToPath(new URL('./test-results/release', import.meta.url));
const traceDir = fileURLToPath(new URL('./test-results/release/traces', import.meta.url));

export default defineConfig({
  testDir: './e2e',
  testMatch: 'release-prompt-matrix.spec.ts',
  outputDir,
  snapshotPathTemplate: '{testDir}/visual-references/v1/{arg}{ext}',
  timeout: 600_000,
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
    viewport: { width: 1600, height: 1200 },
  },
  webServer: [
    {
      command: `uv run --active --no-sync uvicorn gui.server:app --host 127.0.0.1 --port ${apiPort}`,
      cwd: repoRoot,
      env: { ETUDE_GUI_TRACES_DIR: traceDir },
      url: `http://127.0.0.1:${apiPort}/api/traces`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npm run preview -- --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      cwd: frontendDir,
      env: { ETUDE_API_PORT: apiPort },
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
