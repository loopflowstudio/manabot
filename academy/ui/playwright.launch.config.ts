import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
const port = process.env.ACADEMY_LAUNCH_PORT ?? '8037';
const proof = fileURLToPath(new URL('./test-results/launch/addresses.json', import.meta.url));
export default defineConfig({
  testDir: './e2e',
  testMatch: 'configured-launch.spec.ts',
  outputDir: './test-results/launch/browser',
  timeout: 120_000,
  workers: 1,
  expect: { timeout: 30_000 },
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium', reducedMotion: 'reduce' },
  webServer: {
    command: 'uv run --active --no-sync python -m tests.fixtures.configured_launch_server',
    cwd: root,
    env: {
      ACADEMY_LAUNCH_PORT: port,
      ACADEMY_LAUNCH_PROOF: proof,
      ACADEMY_FRONTEND_BUILD: fileURLToPath(new URL('./build', import.meta.url)),
    },
    wait: { stdout: /ACADEMY_LAUNCH_PROOF_READY/ },
    gracefulShutdown: { signal: 'SIGINT', timeout: 15_000 },
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
