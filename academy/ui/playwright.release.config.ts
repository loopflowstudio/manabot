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
  testMatch: ['decision-surface.spec.ts', 'release-prompt-matrix.spec.ts', 'learn.spec.ts', 'combat-presentation.spec.ts', 'choice-navigation.spec.ts', 'elemental-world.spec.ts'],
  outputDir,
  snapshotPathTemplate: '{testDir}/visual-references/v7/{arg}{ext}',
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
      command: `uv run --active --no-sync uvicorn academy.server:app --host 127.0.0.1 --port ${apiPort}`,
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
