import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
const port = process.env.ACADEMY_SEARCH_PORT ?? '8381';
export default defineConfig({
  testDir: './e2e', testMatch: 'search.spec.ts',
  outputDir: './test-results/search/browser', timeout: 60_000, workers: 1,
  expect: { timeout: 15_000 },
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium', reducedMotion: 'reduce',
    viewport: { width: 1280, height: 800 } },
  webServer: {
    command: 'uv run --active --no-sync python -m tests.fixtures.policy_search_server', cwd: root,
    env: { ACADEMY_SEARCH_PORT: port,
      ACADEMY_SEARCH_PROOF: fileURLToPath(new URL('./test-results/search/addresses.json', import.meta.url)),
      ACADEMY_FRONTEND_BUILD: fileURLToPath(new URL('./build', import.meta.url)),
      ACADEMY_CARD_ART: 'off', ACADEMY_PLAY_RECORD_ORIGIN: 'automated_validation',
      OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1' },
    wait: { stdout: /ACADEMY_SEARCH_READY/ },
    gracefulShutdown: { signal: 'SIGINT', timeout: 15_000 }, reuseExistingServer: false, timeout: 60_000,
  },
});
