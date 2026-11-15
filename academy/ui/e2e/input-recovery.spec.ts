import { expect, test, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Command, ExperienceFrame, RecoveryEnvelope } from '../src/lib/types';

const root = fileURLToPath(new URL('../../..', import.meta.url));
const proof = new URL('../test-results/policy/addresses.json', import.meta.url);

for (const lost of ['request', 'reply'] as const) {
  test(`reload reconciles a lost ${lost} using the original Command`, async ({ page }) => {
    const { play } = JSON.parse(readFileSync(proof, 'utf8')) as { play: string };
    let intercept = true;
    let committed = false;
    const sent: Command[] = [];
    const recoveries: RecoveryEnvelope[] = [];
    await page.routeWebSocket('**/ws/play', socket => {
      const server = socket.connectToServer();
      socket.onMessage(raw => {
        const message = JSON.parse(String(raw));
        if (message.type === 'command') {
          sent.push(message.command);
          if (intercept && lost === 'request') {
            intercept = false;
            socket.close();
            server.close();
            return;
          }
        }
        server.send(raw);
      });
      server.onMessage(raw => {
        const message = JSON.parse(String(raw));
        if (message.recovery) recoveries.push(message.recovery);
        if (intercept && lost === 'reply' && message.type === 'command_outcome' && message.status === 'accepted') {
          committed = true;
          intercept = false;
          socket.close();
          server.close();
          return;
        }
        socket.send(raw);
      });
    });
    await page.goto(play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    const initial = recoveries.at(-1)!.frame;
    await page.getByTestId('action-option').first().click();
    await expect.poll(() => intercept).toBe(false);
    const pending = await page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume')!).pending_command);
    expect(pending.command).toEqual(sent[0]);
    expect(committed).toBe(lost === 'reply');
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    await expect.poll(async () => page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume')!).pending_command ?? null)).toBeNull();
    expect(sent.length).toBe(lost === 'request' ? 2 : 1);
    expect(sent.every(command => JSON.stringify(command) === JSON.stringify(sent[0]))).toBe(true);
    const recovered = recoveries.at(-1)!;
    expect(recovered.frame.match_id).toBe(initial.match_id);
    // The ledger proves one commitment, irrespective of later bot/Auto moves.
    const accepted = recovered.accepted_commands.filter(receipt => receipt.command_id === sent[0].command_id);
    if (lost === 'reply') expect(accepted).toHaveLength(1);
    expect(recovered.frame.revision).toBeGreaterThanOrEqual(initial.revision);
  });
}

test('Watch recovers a committed step from its recording after a lost reply', async ({ page }) => {
  const { watch } = JSON.parse(readFileSync(proof, 'utf8')) as { watch: string };
  let steps = 0;
  await page.route('**/api/watch/*/step', async route => {
    steps += 1;
    const reply = await route.fetch();
    expect(reply.ok()).toBe(true);
    if (steps === 1) await route.abort('connectionreset');
    else await route.fulfill({ response: reply });
  });
  await page.goto(watch);
  await page.getByRole('button', { name: 'Start watching' }).click();
  await expect(page.getByText(/^Decision 1 of 1 ·/)).toBeVisible();
  expect(steps).toBe(1);
  await page.getByRole('button', { name: 'Step decision', exact: true }).click();
  await expect(page.getByText(/^Decision 2 of 2 ·/)).toBeVisible();
  expect(steps).toBe(2);
});

async function childServer(port: number): Promise<ChildProcess> {
  const child = spawn('uv', ['run', '--active', '--no-sync', 'python', '-m', 'tests.fixtures.policy_runtime_server'], {
    cwd: root, env: { ...process.env, ACADEMY_POLICY_PORT: String(port),
      ACADEMY_POLICY_PROOF: fileURLToPath(new URL('../test-results/policy/restart/addresses.json', import.meta.url)),
      ACADEMY_FRONTEND_BUILD: fileURLToPath(new URL('../build', import.meta.url)),
      ACADEMY_CARD_ART: 'off', ACADEMY_PLAY_RECORD_ORIGIN: 'automated_validation',
      OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  try {
    await new Promise<void>((resolve, reject) => {
      let output = '';
      const timer = setTimeout(() => reject(new Error(`Recovery server did not start: ${output}`)), 45_000);
      const finish = (error?: Error) => {
        clearTimeout(timer);
        if (error) reject(error); else resolve();
      };
      child.stdout!.on('data', chunk => {
        output += String(chunk);
        if (output.includes('ACADEMY_POLICY_READY')) finish();
      });
      child.stderr!.on('data', chunk => { output += String(chunk); });
      child.once('error', finish);
      child.once('exit', code => finish(new Error(`Recovery server exited (${code}): ${output}`)));
    });
    return child;
  } catch (error) {
    await stopChild(child);
    throw error;
  }
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  await new Promise<void>(resolve => {
    const timer = setTimeout(() => child.kill('SIGKILL'), 10_000);
    child.once('exit', () => { clearTimeout(timer); resolve(); });
    child.kill('SIGINT');
  });
}

for (const theme of ['light', 'dark'] as const) {
  test(`a real backend process restart interrupts the retained game (${theme})`, async ({ page }) => {
    test.setTimeout(120_000);
    await page.setViewportSize(theme === 'light' ? { width: 1280, height: 800 } : { width: 1440, height: 900 });
    await page.addInitScript(theme => localStorage.setItem('etude.theme', theme), theme);
    await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
    const port = Number(process.env.ACADEMY_RECOVERY_PORT ?? 8378);
    let child = await childServer(port);
    try {
      const { play } = JSON.parse(readFileSync(new URL('../test-results/policy/restart/addresses.json', import.meta.url), 'utf8')) as { play: string };
      await page.goto(play);
      await expect(page.getByTestId('action-option').first()).toBeEnabled();
      const frame = await retainedFrame(page);
      await stopChild(child);
      child = await childServer(port);
      await page.reload();
      await expect(page.getByRole('alert')).toContainText('Interrupted');
      expect(await retainedFrame(page)).toBeNull();
      await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(frame!.revision));
      await expect(page.getByTestId('action-option').first()).toBeDisabled();
      expect(await page.evaluate(() => sessionStorage.getItem('academy.launch'))).toBeNull();
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/interrupted-${theme}.png`, import.meta.url)) });
      await page.getByRole('button', { name: 'New Game', exact: true }).focus();
      await page.keyboard.press('Enter');
      // The launch's setup opens first; a second deliberate click starts it.
      const newGame = page.getByRole('button', { name: 'New Game', exact: true });
      if (await page.getByTestId('action-option').first().isDisabled()) await newGame.click();
      await expect(page.getByTestId('action-option').first()).toBeEnabled();
      expect((await retainedFrame(page))!.match_id).not.toBe(frame!.match_id);
    } finally { await stopChild(child); }
  });
}

async function retainedFrame(page: Page): Promise<ExperienceFrame | null> {
  return page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume') ?? 'null')?.last_frame ?? null);
}
