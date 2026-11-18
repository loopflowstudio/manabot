/**
 * The table at laptop sizes. At 1280×800 and 1440×900 the whole game fits the
 * screen: no page scroll, the two battlefields together at least 400 px tall,
 * every card inside its half, and the rail listing every legal offer exactly
 * once. A dense constructed board proves the layout; a seeded real game proves
 * it holds for the boards a game actually reaches.
 */
import { expect, test, type Page } from '@playwright/test';
import boltJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, FrameUpdate, RecoveryEnvelope } from '../src/lib/types';
import { denseFrame } from './fixtures/laptop-table';
import { tableSnapshot } from './fixtures/table';

const VIEWPORTS = [{ width: 1280, height: 800 }, { width: 1440, height: 900 }] as const;

interface Layout {
  scrollY: number; scrollHeight: number; scrollWidth: number; innerHeight: number; innerWidth: number;
  battlefields: number; rows: number; offers: number | null; outside: string[];
}

/**
 * The layout once the board has settled: the hand strip's height eases
 * (`--motion-move`) and the rows resize behind it, so a card is off its half
 * only if it still is after the move.
 */
async function settled(page: Page): Promise<Layout> {
  await expect.poll(async () => (await layout(page)).outside, { timeout: 2_000 }).toEqual([]);
  return layout(page);
}

/** What a person would notice breaking: page scroll, a cramped board, a card off its half, a missing choice. */
async function layout(page: Page): Promise<Layout> {
  return page.evaluate(() => {
    const halves = [...document.querySelectorAll<HTMLElement>('.board .half')];
    const outside: string[] = [];
    // The card a decision is about rises 8 px over its row's edge. An attacker
    // leans toward the midline but stops inside its own half, so it never
    // hides under, or covers, a card of the defending row.
    const slack = 10;
    for (const half of halves) {
      const bounds = half.getBoundingClientRect();
      for (const card of half.querySelectorAll<HTMLElement>('[data-testid=table-card]')) {
        const box = card.getBoundingClientRect();
        if (box.top < bounds.top - slack || box.bottom > bounds.bottom + slack || box.left < bounds.left - slack || box.right > bounds.right + slack) {
          const edges = [box.top - bounds.top, bounds.bottom - box.bottom, box.left - bounds.left, bounds.right - box.right].map(Math.round);
          outside.push(`${card.getAttribute('data-card-name')} inset ${edges.join(' ')} ${card.className}`);
        }
      }
    }
    for (const card of document.querySelectorAll<HTMLElement>('.hand [data-testid=table-card]')) {
      const box = card.getBoundingClientRect();
      if (box.bottom > innerHeight + 1 || box.right > innerWidth + 1) outside.push(`hand ${card.getAttribute('data-card-name')}`);
    }
    const count = document.querySelector('[data-testid=legal-choice-count]')?.textContent?.match(/\d+/)?.[0];
    return {
      scrollY, scrollHeight: document.documentElement.scrollHeight, scrollWidth: document.documentElement.scrollWidth,
      innerHeight, innerWidth, outside,
      battlefields: halves.reduce((sum, half) => sum + half.getBoundingClientRect().height, 0),
      rows: document.querySelectorAll('[data-testid=action-option]').length,
      offers: count === undefined ? null : Number(count),
    };
  });
}

function expectFits(measured: Layout, offers: number): void {
  expect(measured.scrollY).toBe(0);
  expect(measured.scrollHeight).toBeLessThanOrEqual(measured.innerHeight);
  expect(measured.scrollWidth).toBeLessThanOrEqual(measured.innerWidth);
  expect(measured.battlefields).toBeGreaterThanOrEqual(400);
  expect(measured.outside).toEqual([]);
  expect(measured.rows).toBe(offers);
  expect(measured.offers).toBe(offers);
}

/** Serve one dense frame over the play socket; a Command is answered with a stale-revision rejection. */
async function denseTable(page: Page, offers: number): Promise<{ commands: Command[]; frame: ExperienceFrame; resend: () => void }> {
  const frame = denseFrame();
  const mine = frame.projection.agent.battlefield;
  // Many offers over exact visible objects, identical copies included.
  frame.offers = Array.from({ length: offers }, (_, index) => {
    const card = mine[index % mine.length];
    return { ...frame.offers[0], id: 1000 + index, verb: 'activate', label: `Activate ${card.name} ${index + 1}`,
      confirm_label: `Activate ${card.name} ${index + 1}`, focus: [card.id], subjects: [],
      source: { kind: 'object' as const, id: { entity: card.id, incarnation: 0 } } };
  });
  const commands: Command[] = [];
  const recover = (): RecoveryEnvelope => ({ ...boltJson.recovery as RecoveryEnvelope,
    frame, presentation_tail: [], accepted_commands: [], replay_cursor: frame.revision });
  const observe = (): string => JSON.stringify({ type: 'observation', data: frame.projection, actions: [],
    table: tableSnapshot(), session_id: 'laptop', resume_token: 'laptop', recovery: recover() });
  let resend = (): void => {};
  await page.routeWebSocket('**/ws/play', socket => {
    resend = () => socket.send(observe());
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(observe());
      } else if (message.type === 'command') {
        commands.push(message.command);
        frame.revision += 1;
        frame.frame_hash = `laptop-${frame.revision}`;
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: message.command.command_id, code: 'stale_revision', message: 'Inspect the current decision.',
            current_revision: frame.revision, current_prompt: frame.prompt!.id }, recovery: recover() }));
      }
    });
  });
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('table')).toBeVisible();
  return { commands, frame, resend: () => resend() };
}

test('a frame that repeats the revision keeps what the resting pointer shows', async ({ page }) => {
  await page.setViewportSize(VIEWPORTS[1]);
  const { resend, frame } = await denseTable(page, 4);
  const card = page.locator('.board .you [data-testid=table-card]').first();
  await card.hover();
  const preview = page.getByTestId('card-preview');
  await expect(preview).toBeVisible();
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  // The resent frame is told apart by the opponent's life, so the test knows it arrived.
  frame.projection.opponent.life -= 1;
  resend();
  await expect(page.locator('[data-plate-side=opponent] .numeral')).toHaveText(String(frame.projection.opponent.life));
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', revision!);
  await expect(preview).toBeVisible();
});

for (const viewport of VIEWPORTS) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`dense board fits the table ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const { commands, frame } = await denseTable(page, 40);
      await page.evaluate(() => document.fonts.ready);
      await expect(page.getByTestId('action-option')).toHaveCount(40);
      expectFits(await settled(page), 40);
      // Every offer's card is on the board, outlined as acting.
      for (const id of new Set(frame.offers.map(offer => offer.focus[0]))) {
        await expect(page.locator(`.board [data-testid=table-card][data-object-id="${id}"]`).first()).toBeVisible();
      }
      expect(commands).toEqual([]);
      await page.screenshot({ path: testInfo.outputPath('dense.png') });

      // One row is one Command, bound to the decision it was read from; a stale
      // rejection draws the current decision with nothing scrolled.
      const revision = frame.revision;
      const first = frame.offers[0];
      await page.getByTestId('action-option').first().click();
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0]).toMatchObject({ expected_revision: revision, offer_id: first.id, match_id: frame.match_id, prompt_id: frame.prompt!.id });
      await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(revision + 1));
      expectFits(await settled(page), 40);
    });
  }
}

test('a seeded game keeps the table whole at 1280×800', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  let offers = -1;
  let working = false;
  const sent: Command[] = [];
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    server.onMessage(raw => {
      const message = JSON.parse(String(raw)) as { recovery?: RecoveryEnvelope; update?: FrameUpdate; work?: { state: string } | null };
      if ('work' in message) working = message.work?.state === 'running';
      const frame = message.recovery?.frame ?? message.update?.frame;
      if (frame) offers = frame.offers.length;
      socket.send(raw);
    });
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game') message.config.seed = 1;
      if (message.type === 'command') sent.push(message.command);
      server.send(JSON.stringify(message));
    });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  let checked = 0;
  for (let input = 0; input < 60; input += 1) {
    await expect(page.locator('[data-testid=table][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    await expect.poll(() => working).toBe(false);
    if (await page.getByTestId('result-card').isVisible()) break;
    const rows = page.getByTestId('action-option');
    await expect(rows).toHaveCount(offers);
    // A row under the resting pointer previews its payment and lifts the lands it
    // would tap; measure the table at rest.
    await page.mouse.move(640, 4);
    await page.waitForTimeout(250);
    expectFits(await settled(page), offers);
    checked += 1;
    const labels = await rows.allTextContents();
    let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast |Attack with /.test(label));
    if (pick < 0) pick = input % labels.length;
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    const before = sent.length;
    // The pointer travels onto the row before pressing: pointing at a row must
    // not move the rows (a payment note appearing above them once did).
    await rows.nth(pick).click();
    await expect.poll(() => sent.length, { message: `one Command for ${labels[pick]}` }).toBe(before + 1);
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision')), { timeout: 30_000 }).toBeGreaterThan(revision);
  }
  expect(checked).toBeGreaterThanOrEqual(20);
  await page.screenshot({ path: testInfo.outputPath('seeded-game.png') });
});
