import { defineConfig } from '@playwright/test';

const command = process.env.ACADEMY_WHEEL_COMMAND;
const cwd = process.env.ACADEMY_WHEEL_CWD;
if (!command || !cwd) throw new Error('Set ACADEMY_WHEEL_COMMAND and ACADEMY_WHEEL_CWD to an isolated installed wheel');
const port = process.env.ACADEMY_WHEEL_PORT ?? '8027';
export default defineConfig({
  testDir: './e2e',
  testMatch: 'installed-wheel.spec.ts',
  outputDir: './test-results/wheel',
  timeout: 120_000,
  workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium' },
  webServer: {
    command: `${command} --no-open --port ${port} --traces-dir "${cwd}/browser-traces"`,
    cwd,
    env: {
      PATH: `${process.env.ACADEMY_WHEEL_NO_TOOLS ?? ''}:${process.env.PATH ?? ''}`,
      PYTHONPATH: '',
      ACADEMY_FRONTEND_BUILD: '',
    },
    url: `http://127.0.0.1:${port}/api/opponents`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
