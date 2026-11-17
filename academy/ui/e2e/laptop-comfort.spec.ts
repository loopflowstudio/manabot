import { tableSnapshot } from './fixtures/table';
import { writeFile } from 'node:fs/promises';
import { finishBoardNarration } from './presentation-readiness';
import { expect, test, type Locator, type Page } from '@playwright/test';
import boltJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, FrameUpdate, PresentationEvent, RecoveryEnvelope, Trace } from '../src/lib/types';
import type { SharedTableControlEvent, TableSnapshot } from '../src/lib/shared-table-protocol';
import type { RecordedPosition, RecordingIndex } from '../src/lib/recording';
import { denseFrame } from './fixtures/laptop-table';

/** Serve one recorded position through the shared Replay viewer's read API. */
async function routeRecording(page: Page, index: Partial<RecordingIndex> & Pick<RecordingIndex, 'id' | 'players'>,
  position: Partial<RecordedPosition> & Pick<RecordedPosition, 'observation'>): Promise<void> {
  const row = { key: 'latest', kind: 'state' as const, turn: 1, phase: null, step: null, actor: null, automatic: false, can_reveal: false, ...position };
  await page.route(`**/api/traces/${index.id}/recording**`, route => route.fulfill({ json: {
    positions: [row], total: 1, next_offset: null, revision: 1, status: 'finished', can_produce: false, can_study: false,
    winner: null, source_kind: null, watch: false, can_feedback: false, ...index } }));
  await page.route(`**/api/traces/${index.id}/positions/**`, route => route.fulfill({ json: {
    description: null, note_key: `${index.id}:${row.key}`, decision: null, outputs: null, viewer: 0, revealed: false,
    history: [], history_labels: [], history_before: null, cursor: 0, transition_start: 0, consequences_recorded: true, ...row } }));
}

interface TableOptions {
  sparse?: boolean;
  offerCount?: number;
  longInstruction?: boolean;
  holdCommand?: boolean;
  handActions?: number;
}

async function installTable(page: Page, options: TableOptions = {}): Promise<{
  commands: Command[]; frame: ExperienceFrame; rejectPending: () => void; publish: (advance?: boolean) => void;
  control: (message: SharedTableControlEvent) => void; closeSocket: () => void;
}> {
  const frame = denseFrame();
  if (options.sparse) {
    frame.projection.agent.battlefield = frame.projection.agent.battlefield.filter(card => card.types?.is_land).slice(0, 1);
    frame.projection.opponent.battlefield = [];
    // Real open-deck games retain these summaries even with an empty field.
    frame.projection.opponent.main_counts = { 'lea.forest': 40 };
    frame.projection.opponent.unseen_counts = { 'lea.forest': 40 };
    frame.projection.opponent.sideboard_counts = { 'lea.forest': 2 };
    frame.projection.opponent.remaining_sideboard_counts = { 'lea.forest': 2 };
    frame.projection.agent.hand = frame.projection.agent.hand.slice(0, 7);
  }
  // A large published offer set over exact visible objects, including identical copies.
  frame.offers = Array.from({ length: options.offerCount ?? frame.projection.agent.battlefield.length }, (_, index) => {
    const card = frame.projection.agent.battlefield[index % frame.projection.agent.battlefield.length];
    return {
      ...frame.offers[0], id: 1000 + index, label: `Inspect choice ${index + 1}: ${card.name}`,
      focus: [card.id],
    };
  });
  if (options.handActions) {
    const card = frame.projection.agent.hand[0];
    frame.projection.agent.hand[1].name = card.name;
    frame.offers = Array.from({ length: options.handActions }, (_, index) => ({ ...frame.offers[0],
      id: 9000 + index, verb: 'cast', label: `Cast ${card.name} option ${index + 1}`,
      confirm_label: `Cast ${card.name} option ${index + 1}`, focus: [card.id], subjects: [],
      source: { kind: 'object', id: { entity: card.id, incarnation: 0 } } }));
  }
  if (options.longInstruction) {
    frame.prompt!.instruction = `${'Read this instruction before choosing. '.repeat(60)}End of the complete instruction.`;
  }
  let rejectPending = (): void => { throw new Error('No pending command'); };
  let publish = (_advance = true): void => { throw new Error('Socket not ready'); };
  let control = (_message: SharedTableControlEvent): void => { throw new Error('Socket not ready'); };
  let closeSocket = (): void => { throw new Error('Socket not ready'); };
  const commands: Command[] = [];
  await page.routeWebSocket('**/ws/play', socket => {
    const recover = (): RecoveryEnvelope => ({ ...boltJson.recovery as RecoveryEnvelope,
      frame, presentation_tail: [], accepted_commands: [], replay_cursor: frame.revision });
    control = message => socket.send(JSON.stringify(message));
    closeSocket = () => { void socket.close({ code: 4101, reason: 'Controlled disconnect' }); };
    publish = (advance = true) => {
      if (advance) { frame.revision += 1; frame.frame_hash = `changed-${frame.revision}`; }
      socket.send(JSON.stringify({ type: 'observation', data: frame.projection, actions: [], recovery: recover() }));
    };
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({ type: 'observation', data: frame.projection, actions: [],
          table: tableSnapshot(), session_id: 'laptop', resume_token: 'laptop', recovery: recover() }));
      } else if (message.type === 'set_stops') {
        frame.stops = { ...message.stops, auto_pass: message.auto_pass, stop_on_stack: message.stop_on_stack };
        publish();
      } else if (message.type === 'command') {
        commands.push(message.command);
        rejectPending = () => {
          frame.revision += 1;
          frame.frame_hash = `laptop-${frame.revision}`;
          socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
            rejection: { command_id: message.command.command_id, code: 'stale_revision',
              message: 'Inspect the current decision.', current_revision: frame.revision,
              current_prompt: frame.prompt!.id }, recovery: recover() }));
        };
        if (!options.holdCommand) rejectPending();
      }
    });
  });
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
  return { commands, frame, rejectPending: () => rejectPending(), publish: (advance = true) => publish(advance), control: message => control(message), closeSocket: () => closeSocket() };
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`laptop choices, inspection and recovery ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const { commands, frame } = await installTable(page);
      await page.evaluate(() => document.fonts.ready);
      const geometry = await page.evaluate(() => {
        const bounds = (selector: string): DOMRect => document.querySelector(selector)!.getBoundingClientRect();
        return { width: innerWidth, height: innerHeight, dpr: devicePixelRatio,
          scale: visualViewport!.scale, scroll: scrollY, documentWidth: document.documentElement.scrollWidth,
          workspace: bounds('[data-testid="play-workspace"]').toJSON(),
          hand: bounds('[aria-label="You hand"]').toJSON(),
          choices: bounds('#table-choices').toJSON() };
      });
      expect(geometry.scale).toBe(1);
      expect(geometry.scroll).toBe(0);
      expect(geometry.documentWidth).toBe(viewport.width);
      for (const box of [geometry.workspace, geometry.hand, geometry.choices]) {
        expect(box.top).toBeGreaterThanOrEqual(0);
        expect(box.bottom).toBeLessThanOrEqual(viewport.height);
      }
      expect(geometry.choices.left).toBeGreaterThan(geometry.hand.right);
      await writeFile(testInfo.outputPath('geometry.json'), JSON.stringify(geometry, null, 2));
      await page.screenshot({ path: testInfo.outputPath('initial-table.png') });

      const filter = page.getByRole('searchbox', { name: 'Find an action' });
      for (const offer of frame.offers) {
        await filter.fill(offer.label);
        const option = page.getByTestId('action-option').filter({ hasText: offer.label });
        await expect(option).toHaveCount(1);
        await option.focus();
        // Every exact copy remains reachable, even on another battlefield page.
        await expect(page.locator(`[data-object-id="${offer.focus[0]}"]`)).toBeVisible();
      }
      await filter.fill('nothing matches');
      await filter.press('Escape');
      await expect(page.getByTestId('action-option')).toHaveCount(frame.offers.length);
      expect(commands).toEqual([]);
      // Keyboard card inspection changes navigation only.
      const hand = page.getByRole('region', { name: 'You hand', exact: true });
      await hand.getByRole('button').last().focus();
      const handBounds = (await hand.boundingBox())!;
      const lastCard = (await hand.getByRole('button').last().boundingBox())!;
      expect(lastCard.x + lastCard.width).toBeLessThanOrEqual(handBounds.x + handBounds.width);
      await hand.getByRole('button').first().focus();
      await hand.getByRole('button').first().press('Enter');
      await expect(page.getByRole('region', { name: 'Inspected card' })).toBeVisible();
      expect(commands).toEqual([]);
      await page.getByRole('button', { name: 'Show all', exact: false }).click();
      const revision = frame.revision;
      const first = frame.offers[0];
      await page.getByTestId('action-option').first().click();
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0]).toMatchObject({ expected_revision: revision, offer_id: first.id,
        match_id: frame.match_id, prompt_id: frame.prompt!.id });
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(revision + 1));
      expect(await page.evaluate(() => scrollY)).toBe(0);
      const recoveredHand = await hand.boundingBox();
      expect(recoveredHand!.y + recoveredHand!.height).toBeLessThanOrEqual(viewport.height);
      await page.screenshot({ path: testInfo.outputPath('laptop-table.png') });
    });
  }
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const sparse of [true, false]) {
    test(`battlefield lane access ${sparse ? 'sparse' : 'dense'} ${viewport.width}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      const { commands, frame } = await installTable(page, { sparse });
      await page.evaluate(() => document.fonts.ready);
      const zones = page.getByRole('region', { name: 'You battlefield and zones', exact: true });
      const zoneBounds = (await zones.boundingBox())!;
      if (sparse) {
        const land = (await zones.locator('[data-object-id]').first().boundingBox())!;
        expect(land.y).toBeGreaterThanOrEqual(zoneBounds.y);
        expect(land.y + land.height).toBeLessThanOrEqual(zoneBounds.y + zoneBounds.height);
        expect(await zones.evaluate(element => element.scrollTop)).toBe(0);
        const opponentZones = page.locator('[aria-label$="battlefield and zones"]').first();
        await opponentZones.focus();
        await opponentZones.press('End');
        await expect.poll(() => opponentZones.evaluate(element => element.scrollTop)).toBeGreaterThan(0);
        const opponentBounds = (await opponentZones.boundingBox())!;
        await expect.poll(async () => {
          const graveyard = (await opponentZones.locator('summary').filter({ hasText: /^Graveyard/ }).boundingBox())!;
          return graveyard.y + graveyard.height;
        }).toBeLessThanOrEqual(opponentBounds.y + opponentBounds.height);
        expect(await page.evaluate(() => scrollY)).toBe(0);
      }
      const lanes = zones.locator('[data-testid="battlefield"] [data-battlefield-lane]');
      for (const lane of await lanes.all()) {
        const heading = (await lane.locator('h3').boundingBox())!;
        expect(heading.y).toBeGreaterThanOrEqual(zoneBounds.y);
        expect(heading.y + heading.height).toBeLessThanOrEqual(zoneBounds.y + zoneBounds.height);
        const seen = new Set<string>();
        const next = lane.getByRole('button', { name: /^Next / });
        do {
          for (const card of await lane.locator('[data-object-id]').all()) {
            seen.add((await card.getAttribute('data-object-id'))!);
            await card.focus();
            const cardBounds = (await card.boundingBox())!;
            const visibleZone = (await zones.boundingBox())!;
            expect(cardBounds.y).toBeGreaterThanOrEqual(visibleZone.y);
            expect(cardBounds.y + cardBounds.height).toBeLessThanOrEqual(visibleZone.y + visibleZone.height);
          }
          if (!await next.count() || !await next.isEnabled()) break;
          await next.focus();
          await expect(next).toBeInViewport();
          await next.press('Enter');
        } while (true);
        const kind = await lane.getAttribute('data-battlefield-lane');
        const expected = frame.projection.agent.battlefield.filter(card => kind === 'lands' ? card.types?.is_land : card.types?.is_creature);
        expect([...seen].sort()).toEqual(expected.map(card => String(card.id)).sort());
      }
      if (sparse) {
        await expect(page.getByRole('region', { name: 'Search 64 battlefield and zones', exact: true }).getByText('Creatures · 0', { exact: true })).toBeVisible();
      }
      expect(await page.evaluate(() => scrollY)).toBe(0);
      expect(commands).toEqual([]);
      await page.screenshot({ path: testInfo.outputPath('battlefield.png') });
      await writeFile(testInfo.outputPath('battlefield-geometry.json'), JSON.stringify(await zones.evaluate(element => ({
        bounds: element.getBoundingClientRect().toJSON(), scrollTop: element.scrollTop, scrollHeight: element.scrollHeight,
        lanes: Array.from(element.querySelectorAll('[data-battlefield-lane]'), lane => ({ label: lane.getAttribute('aria-label'), bounds: lane.getBoundingClientRect().toJSON() })),
      })), null, 2));
    });
  }
}

async function expectInsideViewport(locator: Locator): Promise<void> {
  await expect(locator).toBeVisible();
  const bounds = (await locator.boundingBox())!;
  const viewport = locator.page().viewportSize()!;
  expect(bounds.y).toBeGreaterThanOrEqual(0);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`persistent context through unfiltered browsing ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const { commands, frame, rejectPending } = await installTable(page, {
        offerCount: 100, longInstruction: true, holdCommand: true,
      });
      const offers = page.getByTestId('action-option');
      const list = page.getByRole('group', { name: 'Legal choices', exact: true });
      const prompt = page.getByTestId('decision-prompt');
      const filter = page.getByRole('searchbox', { name: 'Find an action' });
      const count = page.getByTestId('legal-choice-count');
      const pass = page.getByTestId('pass-turn');
      const automation = page.getByTestId('automation-status');
      const assertRegions = async (): Promise<void> => {
        const boxes = await page.evaluate(() => {
          const box = (selector: string): DOMRect => document.querySelector(selector)!.getBoundingClientRect();
          return { status: box('[data-testid="current-decision"]').toJSON(),
            decision: box('[data-testid="decision-surface"]').toJSON(),
            controls: box('.decision-controls').toJSON(), choices: box('.offer-list').toJSON(),
            automation: box('[data-testid="automation-status"]').toJSON(),
            secondary: box('.secondary-panels').toJSON(),
            rail: box('#table-choices').toJSON(), board: box('#table-board').toJSON() };
        });
        expect(boxes.status.bottom).toBeLessThanOrEqual(boxes.decision.top);
        expect(boxes.controls.bottom).toBeLessThanOrEqual(boxes.choices.top);
        expect(boxes.choices.bottom).toBeLessThanOrEqual(boxes.decision.bottom);
        expect(boxes.decision.bottom).toBeLessThanOrEqual(boxes.automation.top);
        expect(boxes.automation.bottom).toBeLessThanOrEqual(boxes.secondary.top);
        expect(boxes.secondary.bottom).toBeLessThanOrEqual(boxes.rail.bottom);
        expect(boxes.board.top).toBeGreaterThanOrEqual(0);
        expect(boxes.board.bottom).toBeLessThanOrEqual(viewport.height);
        expect(await page.evaluate(() => scrollY)).toBe(0);
      };
      // Browse the full list with the keyboard; never reduce it to one search result.
      await offers.first().focus();
      for (let index = 1; index < frame.offers.length; index += 1) await page.keyboard.press('Tab');
      await expect(offers.last()).toBeFocused();
      for (const control of [prompt, filter, count, pass, automation, offers.last()]) await expectInsideViewport(control);
      expect(await list.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await expect(count).toHaveText('100 of 100 legal choices');
      // All of a long instruction remains independently keyboard-readable.
      await prompt.focus();
      await prompt.press('End');
      await expect.poll(() => prompt.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
      await expectInsideViewport(offers.last());
      expect(commands).toEqual([]);

      // Several offers for an exact copy: Show all must survive last-offer browsing.
      const subject = frame.offers.at(-1)!.focus[0];
      await page.locator(`[data-object-id="${subject}"]`).click();
      const showAll = page.getByRole('button', { name: 'Show all', exact: true });
      await offers.last().focus();
      await expectInsideViewport(showAll);
      await showAll.press('Enter');
      await expect(offers).toHaveCount(100);
      await filter.fill('nothing matches');
      await expect(offers).toHaveCount(0);
      await expectInsideViewport(page.getByRole('button', { name: 'Clear filter', exact: true }));
      await filter.press('Escape');
      await expect(offers).toHaveCount(100);

      const settings = page.getByText('History and settings', { exact: true });
      await settings.click();
      await page.getByTestId('stops-panel').locator('summary').click();
      const secondary = page.getByRole('region', { name: 'History and settings', exact: true });
      await secondary.focus();
      await secondary.press('End');
      await offers.last().focus();
      for (const control of [prompt, count, filter, automation, offers.last()]) await expectInsideViewport(control);
      await assertRegions();
      await page.screenshot({ path: testInfo.outputPath('last-offer-context.png') });
      const revision = frame.revision;
      const last = frame.offers.at(-1)!;
      await offers.last().press('Enter');
      await expect.poll(() => commands.length).toBe(1);
      expect(commands[0]).toMatchObject({ expected_revision: revision, offer_id: last.id,
        match_id: frame.match_id, prompt_id: frame.prompt!.id });
      await expect(offers.last()).toBeDisabled();
      await expectInsideViewport(page.getByText('Actions unavailable', { exact: true }));
      // Recover a different pending frame; navigation cannot retain an old offer or filter.
      frame.prompt!.id += 1;
      frame.prompt!.instruction = 'Recovered pending decision';
      frame.offers = frame.offers.slice(0, 12).map(offer => ({ ...offer, id: offer.id + 2000 }));
      rejectPending();
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(revision + 1));
      await expect(prompt).toHaveText('Recovered pending decision');
      await expect(filter).toHaveValue('');
      await expect(offers).toHaveCount(12);
      await expect(offers.first()).toBeFocused();
      await expectInsideViewport(offers.first());
      await expectInsideViewport(count);
      expect(commands).toHaveLength(1);
      expect(await page.evaluate(() => scrollY)).toBe(0);
      await writeFile(testInfo.outputPath('context-geometry.json'), JSON.stringify(await page.evaluate(() => ({
        viewport: { width: innerWidth, height: innerHeight, scale: visualViewport!.scale, dpr: devicePixelRatio },
        controls: [...document.querySelectorAll('[data-testid="decision-prompt"], [data-testid="legal-choice-count"], .offer-list')]
          .map(node => ({ bounds: node.getBoundingClientRect().toJSON(), scroll: node.scrollTop })),
      })), null, 2));
    });
  }
}

// Actual server Commands supplement the deterministic access fixtures.
for (const [seed, viewport] of [
  [0, { width: 1440, height: 900 }],
  [1, { width: 1280, height: 800 }],
] as const) {
  test(`ordinary seeded laptop game ${seed}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    await page.setViewportSize(viewport);
    const observed = new Map<number, PresentationEvent>();
    let matchId = '';
    let liveFrame: ExperienceFrame | null = null;
    let pendingFrame: { match: string; revision: number; resolve: () => void } | null = null;
    let working = false;
    const submitted: Command[] = [];
    let reloadedTarget = false;
    let selectedSources = 0;
    await page.routeWebSocket('**/ws/play', socket => {
      const server = socket.connectToServer();
      server.onMessage(raw => {
        const message = JSON.parse(String(raw)) as { recovery?: RecoveryEnvelope; update?: FrameUpdate; work?: {state: string} | null };
        if ('work' in message) working = message.work?.state === 'running';
        const frame = message.recovery?.frame ?? message.update?.frame;
        if (frame) { matchId = frame.match_id; liveFrame = frame; }
        if (frame && pendingFrame && frame.match_id === pendingFrame.match && frame.revision > pendingFrame.revision) {
          pendingFrame.resolve();
          pendingFrame = null;
        }
        for (const event of message.recovery?.presentation_tail ?? message.update?.presentation ?? []) observed.set(event.seq, event);
        socket.send(raw);
      });
      socket.onMessage(raw => {
        const message = JSON.parse(String(raw));
        if (message.type === 'new_game') message.config.seed = seed;
        if (message.type === 'command') submitted.push(message.command);
        server.send(JSON.stringify(message));
      });
    });
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByTestId('opponent-select').selectOption('random');
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    await page.getByRole('group', { name: 'Table interaction mode' }).getByRole('button', { name: 'Play', exact: true }).click();
    const receipts: { kind: string | null; label: string | null; scroll: number }[] = [];
    try {
      for (let input = 0; input < 120; input += 1) {
        const options = page.getByTestId('action-option');
        const result = page.getByTestId('game-result');
        await expect(page.locator('[data-testid="action-option"], [data-testid="learn-mode"]:not(:disabled), [data-testid="game-result"]').first()).toBeVisible();
        // An early immutable receipt can precede bot/automation frames. Bind
        // inspection and the next input to the settled, enabled authority.
        await expect.poll(() => !working).toBe(true);
        await expect(page.locator('main')).toHaveAttribute('data-frame-revision', String(liveFrame!.revision));
        if (await result.isVisible()) {
          expect(selectedSources).toBeGreaterThan(0);
          expect(reloadedTarget).toBe(true);
          const sourceEvents = [...observed.values()].sort((left, right) => left.seq - right.seq);
          expect(sourceEvents.length).toBeGreaterThan(0);
          const history = page.getByTestId('presentation-history');
          await page.locator('details.secondary-panels').evaluate((element: HTMLDetailsElement) => { element.open = true; });
          await history.locator('details').evaluate((element: HTMLDetailsElement) => { element.open = true; });
          const sequences = (): Promise<number[]> => history.locator('li[data-presentation-seq]').evaluateAll(rows => rows.map(row => Number(row.getAttribute('data-presentation-seq'))));
          await expect.poll(sequences).toEqual(sourceEvents.slice(-100).map(event => event.seq));
          await page.screenshot({ path: testInfo.outputPath('game-final.png') });
          // The saved transport cursor must not erase history on a fresh page.
          // Reload has no in-memory facts, so it needs the server's retained tail.
          await page.reload();
          await expect(page.getByTestId('game-result')).toBeVisible();
          await page.locator('details.secondary-panels').evaluate((element: HTMLDetailsElement) => { element.open = true; });
          await history.locator('details').evaluate((element: HTMLDetailsElement) => { element.open = true; });
          await expect.poll(sequences).toEqual(sourceEvents.slice(-100).map(event => event.seq));
          await expect(page.getByTestId('presentation-stage')).toHaveCount(0);
          const saved = await page.evaluate(async id => {
            const tokens = JSON.parse(localStorage.getItem('etude.play.record-credentials') ?? '[]') as string[];
            const response = await fetch(`/api/traces/${id}`, { headers: { 'x-etude-participant-tokens': tokens.join(',') } });
            if (!response.ok) throw new Error(`Saved trace unavailable: ${response.status}`);
            return await response.json() as Trace;
          }, matchId);
          expect(saved.events.flatMap(event => event.presentation ?? [])).toEqual(sourceEvents);
          await page.goto(`/replay?trace=${encodeURIComponent(matchId)}`);
          await expect(page.getByRole('slider', { name: 'Replay position' })).toBeVisible();
          const slider = page.getByRole('slider', { name: 'Replay position' });
          await slider.fill((await slider.getAttribute('max'))!);
          await expect(page.getByTestId('game-result-dialog')).toHaveAttribute('role', 'status');
          await history.locator('summary').click();
          const replayHand = page.getByTestId('game-board').locator('.hand-cards').last();
          const handBounds = (await replayHand.boundingBox())!;
          const historyBounds = (await history.boundingBox())!;
          // The score rail's history sits beside the board; neither covers the hand.
          expect(handBounds.x >= historyBounds.x + historyBounds.width || handBounds.y + handBounds.height <= historyBounds.y).toBe(true);
          await expect.poll(sequences).toEqual(sourceEvents.slice(-100).map(event => event.seq));
          await writeFile(testInfo.outputPath('history-source.json'), JSON.stringify({
            matchId, events: sourceEvents.length, kinds: [...new Set(sourceEvents.map(event => event.kind.kind))],
            live: true, saved: true, replay: true, selectedSources, reloadedTarget,
            commands: submitted,
          }, null, 2));
          return;
        }
        // Live offers stay usable while optional narration catches up.
        if (await options.count() === 0) {
          await page.getByTestId('learn-mode').first().click();
          continue;
        }
        expect(await page.evaluate(() => scrollY)).toBe(0);
        const labels = await options.allTextContents();
        let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast /.test(label));
        // This is a workflow proof: commit attacks and decline blocks so an
        // arbitrary input-index cycle cannot prolong a stalled battlefield.
        if (pick < 0) pick = labels.findIndex(label => /Attack with |do not block/.test(label));
        if (pick < 0) pick = input % labels.length;
        const chosen = options.nth(pick);
        receipts.push({ kind: await page.getByTestId('action-panel').getAttribute('data-action-space-kind'),
          label: await chosen.textContent(), scroll: await page.evaluate(() => scrollY) });
        const authority = liveFrame as ExperienceFrame | null;
        expect(authority).not.toBeNull();
        const chosenId = Number(await chosen.getAttribute('data-offer-id'));
        const offer = authority!.offers.find(offer => offer.id === chosenId);
        expect(offer).toBeDefined();
        if (!reloadedTarget && authority!.action_space === 'CHOOSE_TARGET') {
          const identity = { match: authority!.match_id, revision: authority!.revision, hash: authority!.frame_hash, offers: authority!.offers };
          await page.reload();
          await expect(page.getByTestId('connection-badge')).toHaveText('connected');
          await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(identity.revision));
          const recovered = liveFrame as ExperienceFrame | null;
          expect({ match: recovered!.match_id, revision: recovered!.revision, hash: recovered!.frame_hash, offers: recovered!.offers }).toEqual(identity);
          reloadedTarget = true;
        }
        const count = submitted.length;
        const modes = page.getByRole('group', { name: 'Table interaction mode' });
        await modes.getByRole('button', { name: 'Inspect', exact: true }).click();
        if (authority!.action_space !== 'LEARN') await expect(options).toHaveCount(authority!.offers.length);
        await expect(page.getByTestId('legal-choice-count')).toContainText(`of ${authority!.offers.length} legal choices`);
        await modes.getByRole('button', { name: 'Play', exact: true }).click();
        expect(submitted).toHaveLength(count);
        // Synchronize with the authoritative receipt, under the existing test
        // deadline, rather than racing server work against a five-second poll.
        const advanced = new Promise<void>(resolve => {
          pendingFrame = { match: authority!.match_id, revision: authority!.revision, resolve };
        });
        const source = offer!.subjects?.find(subject => subject.role === 'source')?.object.subject ?? offer!.source;
        const playerTarget = offer!.subjects?.find(subject => subject.role === 'target' && subject.object.subject.kind === 'player')?.object.subject;
        if (source?.kind === 'object' && authority!.projection.agent.hand.some(card => card.id === source.id.entity)) {
          await page.locator(`[data-object-id="${source.id.entity}"]`).click();
          expect(submitted).toHaveLength(count);
          await page.locator(`[data-testid="action-option"][data-offer-id="${offer!.id}"]`).click();
          selectedSources += 1;
        } else if (playerTarget?.kind === 'player') {
          await page.getByRole('button', { name: playerTarget.id === authority!.projection.agent.id ? 'Target You' : 'Target Random', exact: true }).click();
          expect(submitted).toHaveLength(count);
          await page.locator(`[data-testid="action-option"][data-offer-id="${offer!.id}"]`).click();
        } else {
          // Reload and mode changes can rebuild the list. Keep the recovered
          // authority's identity rather than its earlier position in the list.
          const commit = page.locator(`[data-testid="action-option"][data-offer-id="${offer!.id}"]`);
          await commit.focus();
          await expect(commit).toBeFocused();
          await commit.press('Enter');
        }
        await advanced;
        expect(submitted).toHaveLength(count + 1);
        expect(submitted.at(-1)).toMatchObject({ match_id: authority!.match_id, expected_revision: authority!.revision,
          prompt_id: authority!.prompt!.id, offer_id: offer!.id });
      }
      throw new Error('Seeded game did not finish within 120 deliberate inputs');
    } finally {
      await writeFile(testInfo.outputPath('game-receipt.json'), JSON.stringify({ seed, viewport, receipts }, null, 2));
      await finishBoardNarration(page);
      await page.screenshot({ path: testInfo.outputPath('workflow-final.png') });
    }
  });
}

test('narrow table keeps explicit board and choices navigation', async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 800 });
  const { commands } = await installTable(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(900);
  await page.getByRole('link', { name: 'Choose a play', exact: true }).click();
  await expect(page.locator('#table-choices')).toBeInViewport();
  await page.getByRole('link', { name: 'Inspect board', exact: true }).click();
  await expect(page.locator('#table-board')).toBeInViewport();
  expect(commands).toEqual([]);
});

test.describe('Play input', () => {
test.use({ actionTimeout: 10_000 });

for (const width of [1440, 1280]) {
  test(`Play selection, explicit commitment and canceled drags ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: width === 1440 ? 900 : 800 });
    await page.emulateMedia({ colorScheme: width === 1280 ? 'light' : 'dark' });
    const { commands, frame } = await installTable(page, { handActions: 1, holdCommand: true });
    const mode = page.getByRole('group', { name: 'Table interaction mode' });
    await mode.getByRole('button', { name: 'Play', exact: true }).click();
    const card = page.locator(`[data-object-id="${frame.projection.agent.hand[0].id}"]`);
    const selected = page.getByRole('region', { name: 'Selected card actions' });
    await card.focus();
    await card.press('Enter');
    await expect(selected).toBeVisible();
    expect(commands).toHaveLength(0);
    await page.getByRole('button', { name: 'Inspect card', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Inspected card' })).toBeVisible();
    await page.getByRole('button', { name: 'Close inspection' }).click();
    await page.keyboard.press('Escape');
    await expect(selected).toHaveCount(0);
    // Same-name sibling is inspectable but cannot borrow this source's action.
    await page.locator(`[data-object-id="${frame.projection.agent.hand[1].id}"]`).click();
    await expect(page.getByTestId('action-option')).toHaveCount(0);
    await card.click();
    const from = (await card.boundingBox())!;
    const drop = (await page.getByTestId('play-drop-area').boundingBox())!;
    await page.mouse.move(from.x + 20, from.y + 20);
    await page.mouse.down();
    await page.mouse.move(drop.x + 20, drop.y + 10, { steps: 3 });
    await expect(page.getByTestId('play-drop-area')).toContainText('Release to Cast');
    await page.keyboard.press('Escape');
    await page.mouse.up();
    await expect(selected).toHaveCount(0);
    expect(commands).toHaveLength(0);
    await page.mouse.move(from.x + 20, from.y + 20);
    await page.mouse.down();
    await page.mouse.move(2, 2, { steps: 3 });
    await page.mouse.up();
    expect(commands).toHaveLength(0);
    await card.focus();
    await card.press('Enter');
    await page.getByText('History and settings', { exact: true }).click();
    await page.getByTestId('presentation-history').locator('summary').click();
    const regions = await page.evaluate(() => {
      const box = (selector: string): DOMRect => document.querySelector(selector)!.getBoundingClientRect();
      return { controls: box('.decision-controls').toJSON(), choices: box('.offer-list').toJSON(),
        automation: box('[data-testid="automation-status"]').toJSON() };
    });
    expect(regions.controls.bottom).toBeLessThanOrEqual(regions.choices.top);
    expect(regions.choices.height).toBeGreaterThanOrEqual(64);
    expect(regions.choices.bottom).toBeLessThanOrEqual(regions.automation.top);
    await page.screenshot({ path: info.outputPath('selected-play.png') });
    await expectInsideViewport(page.getByRole('region', { name: 'You hand', exact: true }));
    await page.getByTestId('action-option').first().press('Enter');
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toMatchObject({ match_id: frame.match_id, expected_revision: frame.revision, prompt_id: frame.prompt!.id, offer_id: 9000 });
    expect(await page.evaluate(() => scrollY)).toBe(0);
    await page.screenshot({ path: info.outputPath('play-table.png') });
  });
}

test('Play drag commits exactly once; ambiguity and changed authority cancel', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const { commands, frame, publish } = await installTable(page, { handActions: 2, holdCommand: true });
  const mode = page.getByRole('group', { name: 'Table interaction mode' });
  await mode.getByRole('button', { name: 'Play', exact: true }).click();
  const card = page.locator(`[data-object-id="${frame.projection.agent.hand[0].id}"]`);
  const drag = async (beforeRelease?: () => Promise<void>): Promise<void> => {
    const from = (await card.boundingBox())!;
    const drop = (await page.getByTestId('play-drop-area').boundingBox())!;
    await page.mouse.move(from.x + 20, from.y + 20);
    await page.mouse.down();
    await page.mouse.move(drop.x + 20, drop.y + 10, { steps: 3 });
    if (beforeRelease) await beforeRelease();
    await page.mouse.up();
  };
  await drag();
  expect(commands).toHaveLength(0);
  await card.click();
  await expect(page.getByTestId('action-option')).toHaveCount(2);
  frame.offers = frame.offers.slice(0, 1);
  publish();
  await expect(page.getByRole('region', { name: 'Selected card actions' })).toHaveCount(0);
  await drag(async () => { publish(); await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(frame.revision)); });
  expect(commands).toHaveLength(0);
  await drag(async () => {
    const inspect = mode.getByRole('button', { name: 'Inspect', exact: true });
    await inspect.focus();
    await inspect.press('Enter');
    await expect(inspect).toHaveAttribute('aria-pressed', 'true');
  });
  expect(commands).toHaveLength(0);
  await mode.getByRole('button', { name: 'Play', exact: true }).click();
  await card.click();
  await page.getByRole('button', { name: 'Edit stops' }).click();
  await expect(page.getByTestId('stops-panel')).toHaveAttribute('open', '');
  await page.getByTestId('auto-pass-toggle').check();
  await expect(page.getByTestId('stop-my-upkeep')).toBeEnabled();
  await page.getByTestId('stop-my-upkeep').click();
  await expect(page.getByRole('region', { name: 'Selected card actions' })).toHaveCount(0);
  expect(commands).toHaveLength(0);
  await page.getByText('History and settings', { exact: true }).click();
  const revision = frame.revision;
  await drag();
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({ match_id: frame.match_id, expected_revision: revision, prompt_id: frame.prompt!.id, offer_id: 9000 });
});

for (const invalidation of ['capability', 'Study', 'disconnect', 'match', 'same-frame recovery', 'pointercancel'] as const) {
  test(`Play drag canceled by ${invalidation}`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    const { frame, commands, publish, control, closeSocket } = await installTable(page, { handActions: 1, holdCommand: true });
    await page.getByRole('group', { name: 'Table interaction mode' }).getByRole('button', { name: 'Play', exact: true }).click();
    const card = page.locator(`[data-object-id="${frame.projection.agent.hand[0].id}"]`);
    const from = (await card.boundingBox())!;
    const drop = (await page.getByTestId('play-drop-area').boundingBox())!;
    await page.mouse.move(from.x + 20, from.y + 20);
    await page.mouse.down();
    await page.mouse.move(drop.x + 20, drop.y + 10, { steps: 3 });
    await expect(page.getByTestId('play-drop-area')).toContainText('Release to Cast');
    if (invalidation === 'capability') {
      const table: TableSnapshot = { contract: 'testing-house-v1', table_id: 'gesture', table_revision: 1, mode: 'live',
        access: { identity: { viewer_id: 'viewer', table_id: 'gesture', rules_viewer: 0 }, role: 'watcher',
          capabilities: ['view_table'], grant_revision: 1 }, participants: [], beliefs: [], decisions: [],
        opponent_label: null, watcher_invite: null };
      control({ type: 'role_changed', table });
    } else if (invalidation === 'Study') {
      control({ type: 'decision_restored', address: 'gesture-study', restored: {
        address: 'gesture-study', ordinal: 0, viewer: 0, revision: frame.revision, presentation_cursor: 0,
        frame, offer: frame.offers[0], continuation: [], command: { command_id: 'recorded', match_id: frame.match_id,
          expected_revision: frame.revision, prompt_id: frame.prompt!.id, offer_id: frame.offers[0].id, answers: [] } } });
    } else if (invalidation === 'disconnect') {
      closeSocket();
      await expect(page.getByTestId('connection-badge')).toHaveText('disconnected');
    } else if (invalidation === 'pointercancel') {
      await card.dispatchEvent('pointercancel', { pointerId: 1 });
    } else {
      if (invalidation === 'match') frame.match_id = 'replacement-match';
      publish(invalidation !== 'same-frame recovery');
    }
    await expect(page.getByRole('region', { name: 'Selected card actions' })).toHaveCount(0);
    await page.mouse.up();
    await expect(page.getByRole('region', { name: 'Selected card actions' })).toHaveCount(0);
    expect(commands).toHaveLength(0);
    if (invalidation === 'capability' || invalidation === 'Study') {
      await card.focus();
      await card.press('Enter');
      await expect(page.getByRole('region', { name: 'Inspected card' })).toBeVisible();
      expect(commands).toHaveLength(0);
    }
  });
}

for (const zone of ['stack', 'private'] as const) {
  test(`Inspect ${zone} cards retains choice navigation`, async ({ page }) => {
    const { frame, commands, publish } = await installTable(page, { handActions: 1 });
    const card = { ...frame.projection.agent.hand[0], id: 80000, zone: 'STACK' };
    if (zone === 'stack') frame.projection.agent.stack = [card];
    else frame.projection.agent.inspected = [card];
    frame.offers = [{ ...frame.offers[0], focus: [card.id], source: { kind: 'object', id: { entity: card.id, incarnation: 0 } } }];
    publish();
    const region = zone === 'stack' ? page.getByRole('region', { name: 'The stack', exact: true }) : page.getByRole('region', { name: 'Private inspection', exact: true });
    await region.getByRole('button').filter({ hasText: card.name }).click();
    await expect(page.getByRole('region', { name: 'Inspected card', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Show all', exact: true })).toBeVisible();
    expect(commands).toHaveLength(0);
  });
}

});

const checkpointName = 'ci-fix-528/consumer/release-smoke/self-play/etude/6b2c855662254a698e5c7a97356b7930/checkpoints/learner_transitions-8';
const checkpointDigest = '7ac5d442da37abf0838098f8a971a193b9f77d11c094d92688ff35e3d12544dc';

for (const viewport of [{ width: 1280, height: 800 }, { width: 1440, height: 900 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`trained identity fits dense laptop table ${viewport.width} ${colorScheme}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme });
      const { control, frame, publish } = await installTable(page);
      await page.evaluate(() => document.fonts.ready);
      const opponentBar = page.locator(`[data-player-id="${frame.projection.opponent.id}"]`);
      const shortHeight = (await opponentBar.boundingBox())!.height;
      frame.projection.stack_objects = [{ id: 701, label: 'Crashing Wave', kind: 'spell', controller: 1 }];
      const table: TableSnapshot = { contract: 'testing-house-v1', table_id: 'identity', table_revision: 1, mode: 'live',
        access: { identity: { viewer_id: 'viewer', table_id: 'identity', rules_viewer: 0 }, role: 'pilot',
          capabilities: ['view_table', 'submit_live_command', 'configure_match'], grant_revision: 1 },
        participants: [], beliefs: [], decisions: [], watcher_invite: null,
        opponent_label: `${checkpointName} · ${checkpointDigest.slice(0, 10)}`,
        opponent: { name: checkpointName, sha256: checkpointDigest, producer: 'manabot' } };
      control({ type: 'role_changed', table });
      publish();
      await expect(opponentBar).toContainText('ci-fix-528/…/learner_transitions-8');
      await expect(opponentBar).toContainText('7ac5d442da');
      expect((await opponentBar.boundingBox())!.height).toBe(shortHeight);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
      const stack = page.getByTestId('ordered-stack');
      await expect(stack).toBeVisible();
      const heading = (await opponentBar.locator('h2').boundingBox())!;
      expect(heading.y + heading.height).toBeLessThanOrEqual((await stack.boundingBox())!.y);
      const battlefield = page.locator('.player-area').first().locator('.staff [data-object-id]').first();
      await expect(battlefield).toBeVisible();
      const hand = page.getByRole('region', { name: 'You hand', exact: true });
      expect((await hand.boundingBox())!.y + (await hand.boundingBox())!.height).toBeLessThanOrEqual(viewport.height);
      await expect(page.getByTestId('decision-surface')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('trained-identity.png') });
      const disclosure = page.locator('.player-area').first().locator('.player-status summary');
      await disclosure.focus();
      await disclosure.press('Enter');
      const provenance = page.locator('.player-area').first().getByTestId('player-identity');
      await expect(provenance).toContainText(checkpointName);
      await expect(provenance).toContainText(checkpointDigest);
      await expect(provenance).toContainText('Producer: manabot');
      await page.screenshot({ path: testInfo.outputPath('identity-disclosure.png') });
      await disclosure.press('Enter');
      const filter = page.getByRole('searchbox', { name: 'Find an action' });
      for (const offer of frame.offers) {
        await filter.fill(offer.label);
        const option = page.getByTestId('action-option').filter({ hasText: offer.label });
        await option.focus();
        const bounds = (await option.boundingBox())!;
        expect(bounds.y).toBeGreaterThanOrEqual(0);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
      }
      const humanName = 'H'.repeat(60);
      await page.route('**/api/traces', route => route.fulfill({ json: [{ id: 'identity-human' }] }));
      await routeRecording(page, { id: 'identity-human', status: 'interrupted',
        opponent: table.opponent, opponent_checkpoint: { source: 'runs', world: 'w6', model: 'Transformer', step: 8 },
        players: [{ seat: 0, name: humanName, kind: 'human', player_id: 'human', deck: 'ur_lessons', version: null, inference: null },
          { seat: 1, name: checkpointName, kind: 'bot', player_id: `checkpoint:${checkpointDigest}`, deck: 'gw_allies', version: checkpointDigest, inference: null }],
      }, { observation: frame.projection });
      await page.goto('/replay?trace=identity-human');
      const heroBar = page.locator(`[data-player-id="${frame.projection.agent.id}"]`);
      await expect(heroBar.locator('h2')).toContainText(humanName);
      expect((await heroBar.locator('h2').boundingBox())!.height).toBeLessThanOrEqual(28);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
      await page.locator('.player-area').first().locator('.player-status summary').click();
      const replayIdentity = page.locator('.player-area').first().getByTestId('player-identity');
      await expect(replayIdentity).toContainText('world: w6');
      await expect(replayIdentity).toContainText('step: 8');
      const replayDisclosure = page.locator('.player-area').first().locator('.player-status summary');
      await replayDisclosure.focus();
      await replayDisclosure.press('End');
      await expect.poll(() => page.locator('.player-area').first().locator('.player-status').evaluate(el => el.scrollTop)).toBeGreaterThan(0);
      await page.screenshot({ path: testInfo.outputPath('long-human-replay.png') });
    });
  }
}

test('Watch keeps recorded identities attached to their seats under stable seating', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const frame = denseFrame();
  // Seat 2 is acting; seating stays physical, so identities follow seats, not the actor.
  frame.projection.agent.player_index = 1;
  frame.projection.opponent.player_index = 0;
  const players = [
    { seat: 0, player_id: 'reference', name: 'Reference', kind: 'bot' as const, deck: 'gw_allies', version: null, inference: null },
    { seat: 1, player_id: `checkpoint:${checkpointDigest}`, name: checkpointName, kind: 'bot' as const, deck: 'ur_lessons', version: checkpointDigest, inference: null },
  ];
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await routeRecording(page, { id: 'identity-watch', players, watch: true, winner: 1 },
    { key: 'choice:0', kind: 'choice', actor: 1, viewer: 1, observation: frame.projection });
  await page.goto('/replay?watch=1&trace=identity-watch');
  const regions = page.locator('.player-area');
  await expect(page.getByText('Near: seat 1', { exact: true })).toBeVisible();
  await expect(regions.last().locator('h2')).toContainText('Reference');
  await expect(regions.first().locator('h2')).toContainText('ci-fix-528/…/learner_transitions-8');
  await expect(regions.first().locator('h2')).toContainText('7ac5d442da');
  await page.getByRole('button', { name: 'Swap seats' }).click();
  await expect(regions.last().locator('h2')).toContainText('ci-fix-528/…/learner_transitions-8');
  await expect(regions.first().locator('h2')).toContainText('Reference');
  await expect(regions.last().locator('h2 .player-name')).toHaveAttribute('title', `${checkpointName} · ${checkpointDigest}`);
});
