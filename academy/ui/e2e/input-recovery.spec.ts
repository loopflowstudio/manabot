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
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  expect(steps).toBe(1);
  // The read after the lost reply found the new choice. Navigation reads it
  // and its consequences before the frontier can deliberately produce again.
  for (const position of [1, 2, 3]) {
    await page.getByRole('button', {name: 'Next', exact: true}).click();
    await expect(page.getByTestId('position-counter')).toHaveText(`Position ${position} / 3`);
    expect(steps).toBe(1);
  }
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 3 / 5');
  expect(steps).toBe(2);
});

async function childServer(port: number): Promise<ChildProcess> {
  const child = spawn('uv', ['run', '--active', '--no-sync', 'python', '-m', 'tests.fixtures.policy_runtime_server'], {
    cwd: root, env: { ...process.env, ACADEMY_POLICY_PORT: String(port),
      ACADEMY_POLICY_PROOF: fileURLToPath(new URL('../test-results/policy/restart/addresses.json', import.meta.url)),
      ACADEMY_FRONTEND_BUILD: fileURLToPath(new URL('../build', import.meta.url)),
      ACADEMY_CARD_ART: 'off', ACADEMY_PLAY_RECORD_ORIGIN: 'automated_validation',
      OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1' },
    detached: true,
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
  // Abrupt loss exercises authority loss without a graceful recording flush.
  // Kill the owned group so uv and its Python/server descendants are reaped.
  await new Promise<void>(resolve => {
    child.once('exit', () => resolve());
    try { process.kill(-child.pid!, 'SIGKILL'); }
    catch { resolve(); }
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
      const opponent = await page.locator('.player-status').first().textContent();
      expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume')!).identity?.opponent?.name)).toContain('Independent');
      await stopChild(child);
      child = await childServer(port);
      // The live tab must recognize lost authority before any reload.
      await expect(page.getByRole('alert')).toContainText('Interrupted');
      expect(await retainedFrame(page)).toBeNull();
      await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(frame!.revision));
      await expect(page.getByTestId('action-option').first()).toBeDisabled();
      expect(await page.evaluate(() => sessionStorage.getItem('academy.launch'))).toBeNull();
      const snapshot = await page.evaluate(() => JSON.parse(sessionStorage.getItem('academy.interrupted')!));
      expect(snapshot.match_id).toBe(frame!.match_id);
      expect(snapshot.identity?.opponent?.name).toContain('Independent');
      expect(snapshot).not.toHaveProperty('resume_token');
      for (let reload = 0; reload < 3; reload++) {
        if (reload > 0) await page.reload();
        await expect(page.getByTestId('connection-badge')).toHaveText('Interrupted');
        await expect(page.getByTestId('decision-prompt')).toContainText('cannot be submitted');
        await expect(page.getByTestId('legal-choice-count')).toContainText('recorded choices');
        await expect(page.getByTestId('legal-choice-count')).not.toContainText('legal choices');
        await expect(page.getByTestId('pass-turn')).toHaveCount(0);
        await expect(page.getByRole('button', {name: 'Rematch', exact: true})).toHaveCount(0);
        await expect(page.getByRole('alert')).toContainText('Interrupted');
        await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(frame!.revision));
        await expect(page.getByTestId('action-option').first()).toBeDisabled();
        expect(await retainedFrame(page)).toBeNull();
        expect(await page.locator('.player-status').first().textContent()).toBe(opponent);
      }
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/policy/interrupted-${theme}.png`, import.meta.url)) });
      const setup = page.getByRole('banner').getByRole('link', {name: 'New game', exact: true});
      await expect(setup).toBeVisible();
      await setup.focus();
      await page.keyboard.press('Enter');
      await page.getByRole('button', { name: 'New Game', exact: true }).click();
      await expect(page.getByTestId('action-option').first()).toBeEnabled();
      expect((await retainedFrame(page))!.match_id).not.toBe(frame!.match_id);
    } finally { await stopChild(child); }
  });
}

async function retainedFrame(page: Page): Promise<ExperienceFrame | null> {
  return page.evaluate(() => JSON.parse(sessionStorage.getItem('etude.gui.resume') ?? 'null')?.last_frame ?? null);
}

for (const seat of [0, 1]) {
  test(`Watch names loading seat ${seat + 1} and reconciles a lost start reply`, async ({page}) => {
    const addresses = JSON.parse(readFileSync(proof, 'utf8')) as {watch: string; ids: {controlled: string}};
    const url = new URL(addresses.watch);
    url.searchParams.set('seat1', `policy:${addresses.ids.controlled}`);
    url.searchParams.set('seat2', `policy:${addresses.ids.controlled}`);
    const base = url.origin;
    await page.request.post(`${base}/api/__fixtures/watch-load/${seat}/block`);
    let lost = false;
    let producer: {id: string; token: string} | null = null;
    let starts = 0;
    let dropCompleted = true;
    await page.route('**/api/watch/setups/*', async route => {
      if (route.request().method() !== 'POST') {
        const response = await route.fetch();
        const result = await response.json();
        if (dropCompleted && result.created) return route.abort('connectionreset');
        return route.fulfill({response});
      }
      starts++;
      const response = await route.fetch();
      producer = (await response.json()).created;
      lost = true;
      await route.abort('connectionreset');
    });
    try {
      await page.goto(url.href);
      await page.getByRole('button', {name: 'Start watching', exact: true}).click();
      await expect(page.getByTestId('work-status')).toContainText(`Loading bot · Seat ${seat + 1}`);
      await expect(page.getByTestId('work-status')).toContainText('elapsed');
      expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem('academy.watch-setup')!).handle)).toBeTruthy();
      await page.screenshot({path: fileURLToPath(new URL(`../test-results/policy/watch-load-seat-${seat}.png`, import.meta.url))});
      await page.request.post(`${base}/api/__fixtures/watch-load/${seat}/release`);
      await expect.poll(() => lost).toBe(true);
      await expect(page.getByRole('alert')).toBeVisible();
      // Reload retains the start envelope; explicit Start checks it, ignoring
      // mutable form choices, rather than creating another policy pair.
      dropCompleted = false;
      await page.reload();
      await page.getByRole('button', {name: 'Start watching', exact: true}).click();
      await expect(page.getByRole('button', {name: 'Continue', exact: true})).toBeEnabled();
      expect(new URL(page.url()).searchParams.get('trace')).toBe(producer!.id);
      expect(starts).toBe(1);
      expect(await page.evaluate(() => sessionStorage.getItem('academy.watch-setup'))).toBeNull();
    } finally {
      await page.request.post(`${base}/api/__fixtures/watch-load/${seat}/release`);
    }
  });
}

for (const theme of ['light', 'dark'] as const) {
  test(`a complete native-policy game retains its final replay (${theme})`, async ({page}) => {
    test.setTimeout(120_000);
    const {watch} = JSON.parse(readFileSync(proof, 'utf8')) as {watch: string};
    const origin = new URL(watch).origin;
    await page.setViewportSize(theme === 'light' ? {width: 1280, height: 800} : {width: 1440, height: 900});
    await page.emulateMedia({colorScheme: theme, reducedMotion: 'reduce'});
    await page.addInitScript(theme => localStorage.setItem('etude.theme', theme), theme);
    // Exercise a real terminal recording without a long Random mirror: the
    // creature deck plays against a passive opponent on one fixed deal.
    const response = await page.request.post(`${origin}/api/watch`, {data: {
      players: ['random', 'passive'], hero_deck: 'gw_allies', villain_deck: 'ur_lessons', seed: 368,
    }});
    expect(response.ok()).toBe(true);
    const created = await response.json() as {id: string; token: string};
    const headers = {'x-etude-participant-token': created.token};
    let revision = 0;
    let finished = false;
    while (!finished && revision < 128) {
      const step = await page.request.post(`${origin}/api/watch/${created.id}/step`, {headers, data: {revision}});
      expect(step.ok()).toBe(true);
      const result = await step.json() as {revision: number; finished: boolean};
      expect(result.revision).toBe(revision + 1);
      revision = result.revision; finished = result.finished;
    }
    expect(finished).toBe(true);
    const indexResponse = await page.request.get(`${origin}/api/traces/${created.id}/recording`, {headers});
    const index = await indexResponse.json() as {positions: {key: string}[]; next_offset: number | null};
    while (index.next_offset !== null) {
      const next = await (await page.request.get(`${origin}/api/traces/${created.id}/recording?offset=${index.next_offset}`, {headers})).json();
      index.positions.push(...next.positions); index.next_offset = next.next_offset;
    }
    await page.goto(`${origin}/replay?watch=1`);
    await page.evaluate(created => sessionStorage.setItem(`etude.watch.${created.id}`, created.token), created);
    const last = index.positions.at(-1)!.key;
    await page.goto(`${origin}/replay?trace=${created.id}&position=${encodeURIComponent(last)}`);
    await expect(page.getByTestId('game-result-dialog')).toBeVisible();
    await expect(page.getByRole('button', {name: 'Continue', exact: true})).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({path: fileURLToPath(new URL(`../test-results/policy/complete-replay-${theme}.png`, import.meta.url))});
    await page.getByRole('button', {name: 'Previous', exact: true}).focus();
    await page.keyboard.press('Enter');
    await page.getByRole('button', {name: 'Next', exact: true}).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('game-result-dialog')).toBeVisible();
  });
}

test('historical positions stay separate from blocked frontier work and late source replies', async ({page}) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as {watch: string; ids: {controlled: string}};
  const origin = new URL(addresses.watch).origin;
  const first = await (await page.request.post(`${origin}/api/watch`, {data: {players: [`policy:${addresses.ids.controlled}`, `policy:${addresses.ids.controlled}`], seed: 42}})).json() as {id: string; token: string};
  const other = await (await page.request.post(`${origin}/api/watch`, {data: {players: ['random', 'random'], seed: 1}})).json() as {id: string; token: string};
  await page.goto(`${origin}/replay?watch=1`);
  await page.evaluate(records => { for (const record of records) sessionStorage.setItem(`etude.watch.${record.id}`, record.token); }, [first, other]);
  await page.goto(`${origin}/replay?trace=${first.id}`);
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  const slider = page.getByRole('slider', {name: 'Replay position'});
  await slider.focus(); await page.keyboard.press('End');
  await expect(page.getByTestId('position-counter')).toHaveText('Position 3 / 3');
  for (const seat of [0, 1]) await page.request.post(`${origin}/api/__fixtures/watch-choose/${seat}/block`);
  let releaseStatus: (() => void) | null = null;
  let delivered = false;
  let statusReads = 0;
  await page.route(`**/api/watch/${first.id}/work`, async route => {
    const response = await route.fetch();
    if (++statusReads > 1) { await new Promise<void>(resolve => releaseStatus = resolve); delivered = true; }
    await route.fulfill({response});
  });
  try {
    const pendingStep = page.waitForResponse(response => response.url().endsWith(`/api/watch/${first.id}/step`));
    await page.getByRole('button', {name: 'Continue', exact: true}).click();
    await expect(page.getByTestId('work-status')).toContainText('Choosing bot action');
    await page.getByRole('button', {name: 'Previous', exact: true}).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
    await expect(page.getByRole('region', {name: 'Live frontier work'})).toContainText('Reading a saved position; bot work belongs to the frontier.');
    await expect.poll(() => releaseStatus !== null).toBe(true);
    await page.screenshot({path: fileURLToPath(new URL('../test-results/policy/historical-frontier.png', import.meta.url))});
    await page.getByTestId('trace-select').selectOption(other.id);
    await expect(page.getByTestId('trace-select')).toHaveValue(other.id);
    for (const seat of [0, 1]) await page.request.post(`${origin}/api/__fixtures/watch-choose/${seat}/release`);
    releaseStatus?.();
    await pendingStep;
    await expect.poll(() => delivered).toBe(true);
    await expect(page.getByTestId('work-status')).toHaveCount(0);
    await expect(page.getByTestId('trace-select')).toHaveValue(other.id);
  } finally {
    releaseStatus?.();
    for (const seat of [0, 1]) await page.request.post(`${origin}/api/__fixtures/watch-choose/${seat}/release`);
  }
});
