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
    expect((await tabState(page))!.pending_command!.command).toEqual(sent[0]);
    expect(committed).toBe(lost === 'reply');
    await page.reload();
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    await expect.poll(async () => (await tabState(page))?.pending_command ?? null).toBeNull();
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

/** What this tab keeps for the game on its board: the Command in flight and the last frame. */
async function tabState(page: Page): Promise<{ pending_command?: { command: Command } | null; last_frame?: ExperienceFrame } | null> {
  return page.evaluate(() => {
    const game = document.querySelector<HTMLElement>('main')?.dataset.game;
    return game ? JSON.parse(sessionStorage.getItem(`academy.tab.${game}`) ?? 'null') : null;
  });
}

/** Make the first offered choice and wait for the next decision. */
async function decide(page: Page): Promise<void> {
  const option = page.getByTestId('action-option').first();
  await expect(option).toBeEnabled({ timeout: 30_000 });
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  await option.click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revision!, { timeout: 30_000 });
  await expect(option).toBeEnabled({ timeout: 30_000 });
}

/** Start a game against the scripted passive opponent from the setup screen; returns its id. */
async function startPassive(page: Page, origin: string, address = '/?hero=ur_lessons&villain=gw_allies'): Promise<string> {
  await page.goto(`${origin}${address}`);
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 30_000 });
  return (await page.locator('main').getAttribute('data-game'))!;
}

test('a game survives closing the tab and restarting the server, and resumes from History', async ({ context, page }) => {
  test.setTimeout(180_000);
  const port = Number(process.env.ACADEMY_RECOVERY_PORT ?? 8378);
  let child = await childServer(port);
  try {
    const { play } = JSON.parse(readFileSync(new URL('../test-results/policy/restart/addresses.json', import.meta.url), 'utf8')) as { play: string };
    const origin = new URL(play).origin;
    await page.goto(play);
    await expect(page.getByTestId('action-option').first()).toBeEnabled({ timeout: 30_000 });
    for (let decision = 0; decision < 3; decision++) await decide(page);
    const game = (await page.locator('main').getAttribute('data-game'))!;
    expect(game).not.toBe('');
    // The address names the game, so a reload or a bookmark returns to it.
    expect(new URL(page.url()).searchParams.get('game')).toBe(game);
    const frame = (await tabState(page))!.last_frame!;
    const history = await page.getByTestId('game-log').getByTestId('log-entry').allInnerTexts();
    expect(history.length).toBeGreaterThanOrEqual(3);
    const opponent = await page.locator('[data-testid=player-plate][data-plate-side=opponent] [data-testid=player-target]').textContent();
    expect(opponent).toContain('Independent');
    await page.close();
    // Abrupt loss: nothing is flushed on the way down, and the new process holds no game in memory.
    await stopChild(child);
    child = await childServer(port);

    const returned = await context.newPage();
    await returned.goto(`${origin}/games`);
    const row = returned.getByTestId('game-history-row').filter({ has: returned.locator(`a[aria-label="Resume game ${game}"]`) });
    await expect(row.getByTestId('game-history-status')).toHaveText('In progress');
    await row.getByTestId('game-history-resume').click();
    await expect(returned.getByTestId('action-option').first()).toBeEnabled({ timeout: 60_000 });
    // The same game, at the decision that was pending, with its history and its opponent.
    await expect(returned.locator('main')).toHaveAttribute('data-game', game);
    // Taking the game up from its record is a new revision of the same decision.
    await expect(returned.locator('main')).toHaveAttribute('data-frame-revision', String(frame.revision + 1));
    const resumed = (await tabState(returned))!.last_frame!;
    expect(resumed.match_id).toBe(frame.match_id);
    expect(resumed.offers.map(offer => offer.label)).toEqual(frame.offers.map(offer => offer.label));
    expect(await returned.getByTestId('game-log').getByTestId('log-entry').allInnerTexts()).toEqual(history);
    expect(await returned.locator('[data-testid=player-plate][data-plate-side=opponent] [data-testid=player-target]').textContent()).toBe(opponent);
    await returned.screenshot({ path: fileURLToPath(new URL('../test-results/policy/resumed.png', import.meta.url)) });
    await decide(returned);
    // `/` alone opens the game in progress.
    await returned.goto(`${origin}/`);
    await expect(returned.locator('main')).toHaveAttribute('data-game', game, { timeout: 60_000 });
    await expect(returned.getByTestId('action-option').first()).toBeEnabled({ timeout: 30_000 });

    // An unfinished game replays like any other: every recorded decision, and its line on request.
    await returned.goto(`${origin}/replay?trace=${encodeURIComponent(game)}`);
    await expect(returned.getByTestId('game-board')).toBeVisible({ timeout: 30_000 });
    const moves = returned.getByTestId('game-log').locator('button[data-move]');
    expect(await moves.count()).toBeGreaterThanOrEqual(4);
    await moves.first().click();
    await expect(moves.first()).toHaveAttribute('aria-current', 'true');
    await moves.last().click();
    await expect(moves.last()).toHaveAttribute('aria-current', 'true');
    await returned.getByTestId('game-menu').click();
    await expect(returned.getByTestId('replay-resume')).toHaveAttribute('href', `/?game=${encodeURIComponent(game)}`);
  } finally { await stopChild(child); }
});

test('two tabs on two games each keep their own Command in flight', async ({ context, page }) => {
  const origin = new URL((JSON.parse(readFileSync(proof, 'utf8')) as { play: string }).play).origin;
  const other = await context.newPage();
  const first = await startPassive(page, origin);
  // Bare `/` would take the first tab's game; an address that names a matchup opens setup beside it.
  const second = await startPassive(other, origin, '/?hero=gw_allies&villain=ur_lessons');
  expect(first).not.toBe(second);
  await decide(page);
  await decide(other);
  // Each tab keeps its own frame under its own game; only the pointer is shared.
  expect((await tabState(page))!.last_frame!.match_id).not.toBe((await tabState(other))!.last_frame!.match_id);
  expect(await page.evaluate(game => sessionStorage.getItem(`academy.tab.${game}`), second)).toBeNull();
  await page.reload();
  await expect(page.locator('main')).toHaveAttribute('data-game', first, { timeout: 30_000 });
  await other.reload();
  await expect(other.locator('main')).toHaveAttribute('data-game', second, { timeout: 30_000 });

  // A third tab at bare `/` leaves the remembered game with the tab that holds it.
  const third = await context.newPage();
  await third.goto(`${origin}/`);
  await expect(third.getByTestId('home')).toBeVisible();
  const takeOver = third.getByTestId('game-open-elsewhere').getByRole('link');
  await expect(takeOver).toHaveAttribute('href', `/?game=${encodeURIComponent(second)}`);
  await decide(other);
  // Its link takes the game over, and the tab that held it says so.
  await takeOver.click();
  await expect(third.locator('main')).toHaveAttribute('data-game', second, { timeout: 30_000 });
  await expect(other.getByText('open in another tab')).toBeVisible();
});

test('session storage failure leaves the game usable with an honest warning', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(window, 'sessionStorage', { get: () => { throw new DOMException('Disabled', 'SecurityError'); } }));
  await startPassive(page, new URL((JSON.parse(readFileSync(proof, 'utf8')) as { play: string }).play).origin);
  await expect(page.getByText(/A reload cannot recover a choice in flight/)).toBeVisible();
  await decide(page);
});

for (const seat of [0, 1]) {
  test(`Watch names loading seat ${seat + 1} and reconciles a lost start reply`, async ({page}) => {
    const addresses = JSON.parse(readFileSync(proof, 'utf8')) as {watch: string; ids: {controlled: string}};
    const url = new URL(addresses.watch);
    url.searchParams.set('seat1', `policy:${addresses.ids.controlled}`);
    url.searchParams.set('seat2', `policy:${addresses.ids.controlled}`);
    const base = url.origin;
    await page.request.post(`${base}/api/__fixtures/watch-load/${seat}/block`);
    let lost = false;
    let producer: {id: string} | null = null;
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
    // Eight-card land-only decks end by native deck-out, independently of
    // random combat choices and automatic mana-payment preferences.
    const response = await page.request.post(`${origin}/api/watch`, {data: {
      players: ['random', 'passive'], seed: 368,
      deck_import: {hero: {source: '8 Island'}, villain: {source: '8 Mountain'}},
    }});
    expect(response.ok()).toBe(true);
    const created = await response.json() as {id: string};
    let revision = 0;
    let finished = false;
    while (!finished && revision < 128) {
      const step = await page.request.post(`${origin}/api/watch/${created.id}/step`, {data: {revision}});
      expect(step.ok()).toBe(true);
      const result = await step.json() as {revision: number; finished: boolean};
      expect(result.revision).toBe(revision + 1);
      revision = result.revision; finished = result.finished;
    }
    expect(finished).toBe(true);
    const indexResponse = await page.request.get(`${origin}/api/traces/${created.id}/recording`);
    const index = await indexResponse.json() as {positions: {key: string}[]; next_offset: number | null};
    while (index.next_offset !== null) {
      const next = await (await page.request.get(`${origin}/api/traces/${created.id}/recording?offset=${index.next_offset}`)).json();
      index.positions.push(...next.positions); index.next_offset = next.next_offset;
    }
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
  const first = await (await page.request.post(`${origin}/api/watch`, {data: {players: [`policy:${addresses.ids.controlled}`, `policy:${addresses.ids.controlled}`], seed: 42}})).json() as {id: string};
  const other = await (await page.request.post(`${origin}/api/watch`, {data: {players: ['random', 'random'], seed: 1}})).json() as {id: string};
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
    // Open another recording in-app (a client navigation, as a link in Games is) while the step is pending.
    await page.evaluate(id => {
      const link = Object.assign(document.createElement('a'), {href: `/replay?trace=${id}`});
      document.body.append(link);
      link.click();
    }, other.id);
    await expect.poll(() => new URL(page.url()).searchParams.get('trace')).toBe(other.id);
    for (const seat of [0, 1]) await page.request.post(`${origin}/api/__fixtures/watch-choose/${seat}/release`);
    releaseStatus?.();
    await pendingStep;
    await expect.poll(() => delivered).toBe(true);
    await expect(page.getByTestId('work-status')).toHaveCount(0);
    expect(new URL(page.url()).searchParams.get('trace')).toBe(other.id);
  } finally {
    releaseStatus?.();
    for (const seat of [0, 1]) await page.request.post(`${origin}/api/__fixtures/watch-choose/${seat}/release`);
  }
});
