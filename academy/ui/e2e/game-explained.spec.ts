/**
 * End-to-end game explanation proof (MTG-391):
 * - A real game's history indents consequences with their rule ("lethal damage"),
 *   and selecting lines draws their kept boards (cast/damage/death) while
 *   naming the cursor in the top bar.
 * - Opening the saved recording in Review draws the same history and steps into
 *   per-line boards retained on `TraceEvent.boards`.
 */
import { expect, test, type Page } from '@playwright/test';
import { selectDeck } from './deck-selection';
import { showEachCreaturesChoice } from './fixtures/combat';
import { gameReply } from './fixtures/game';
import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { PresentationEvent, RecoveryEnvelope } from '../src/lib/types';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

async function playToDeath(page: Page): Promise<boolean> {
  for (let i = 0; i < 100; i += 1) {
    await expect(page.locator('[data-testid=game][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    const deathLines = page.locator('[data-testid=game-log] [data-testid=history-consequence]').filter({ hasText: /died · lethal damage/ });
    if (await deathLines.count() > 0) return true;
    if (await page.getByTestId('result-card').isVisible()) break;
    await showEachCreaturesChoice(page);
    const rows = page.getByTestId('action-option');
    const labels = await rows.allTextContents();
    // Prefer casting burn/creatures, targeting creatures, and attacking/blocking so combat or burn kills a creature.
    let pick = labels.findIndex(l => /Keep hand|Play (Island|Mountain)|Cast Lightning Bolt|Cast (Raging Goblin|Gray Ogre|Wind Drake)|Target (Gray Ogre|Wind Drake|Raging Goblin)|Attack with |Block /.test(l));
    if (pick < 0) pick = labels.findIndex(l => /Pass priority|Decline|do not/.test(l));
    const revision = Number(await page.locator('main').getAttribute('data-frame-revision'));
    await rows.nth(Math.max(0, pick)).click();
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-frame-revision')), { timeout: 30_000 }).toBeGreaterThan(revision);
  }
  const deathLines = page.locator('[data-testid=game-log] [data-testid=history-consequence]').filter({ hasText: /died · lethal damage/ });
  return (await deathLines.count()) > 0;
}

test('indented consequences, per-line kept boards, and Review TraceEvent.boards agree on a real game', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = 11;
      server.send(JSON.stringify(payload));
    });
  });

  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();

  const sawDeath = await playToDeath(page);
  expect(sawDeath).toBe(true);

  // Expand any collapsed busy-turn consequences so the death line is clickable.
  const toggles = page.locator('[data-testid=consequence-toggle][aria-expanded="false"]');
  for (let i = 0; i < await toggles.count(); i += 1) {
    await toggles.nth(0).click();
  }

  const deathButton = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /died · lethal damage/ }).first();
  await expect(deathButton).toBeVisible();
  const castButton = page.locator('[data-testid=game-log] button[data-history-seq]').filter({ hasText: /Cast Lightning Bolt, targeting Gray Ogre/ }).first();
  const damageButton = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /3 damage to Gray Ogre/ }).first();

  // 1. Cast board: Lightning Bolt on the stack, Gray Ogre still on the opponent's battlefield.
  await castButton.click();
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'step');
  await expect(page.getByTestId('stack-entry').first()).toContainText('Lightning Bolt');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath('live-cast-cursor.png') });

  // 2. Damage board: stepping → shows Lightning Bolt off the stack and Gray Ogre on the battlefield with 3 damage and −3 stop delta.
  await page.keyboard.press('ArrowRight');
  await expect(damageButton).toHaveAttribute('aria-current', 'true');
  await expect(page.getByTestId('stack-entry')).toHaveCount(0);
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  await expect(page.locator('.half.opponent [data-testid=stop-delta][data-delta-kind=damage]').first()).toHaveText('−3');
  await expect(page.getByTestId('walk-caption')).toContainText('3 to Gray Ogre');
  await page.screenshot({ path: testInfo.outputPath('live-damage-cursor.png') });

  // 3. Death board (reduced motion): Gray Ogre gone, rule chip reads where it was with ZERO death travel.
  await page.keyboard.press('ArrowRight');
  await expect(deathButton).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(0);
  await expect(page.getByTestId('death-travel')).toHaveCount(0);
  await expect(page.getByTestId('death-chip').first()).toContainText('lethal damage');
  await expect(page.getByTestId('turn-number')).toContainText('Your Lightning Bolt · Gray Ogre died · lethal damage');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await expect(page.getByTestId('recent')).toHaveCount(0);
  await expect(page.getByTestId('walk-caption')).toContainText('lethal damage');
  await page.screenshot({ path: testInfo.outputPath('live-death-cursor.png') });

  // Esc returns to now.
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('game-board')).toHaveAttribute('data-board', 'frame');
  await expect(page.getByTestId('turn-number')).not.toContainText('Esc for now');

  // The address names the game on the board; open it in Review.
  const recordingId = new URL(page.url()).searchParams.get('game');
  expect(recordingId).toBeTruthy();

  await page.goto(`/replay?trace=${encodeURIComponent(recordingId!)}&review=1`);
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 30_000 });

  // Expand busy turns in Review if any.
  const reviewToggles = page.locator('[data-testid=consequence-toggle][aria-expanded="false"]');
  for (let i = 0; i < await reviewToggles.count(); i += 1) {
    await reviewToggles.nth(0).click();
  }
  const reviewCast = page.locator('[data-testid=game-log] button[data-move]').filter({ hasText: /Cast Lightning Bolt, targeting Gray Ogre/ }).first();
  const reviewDamage = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-move]').filter({ hasText: /3 damage to Gray Ogre/ }).first();
  const reviewDeath = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-move]').filter({ hasText: /died · lethal damage/ }).first();
  // Selecting damage inside the consequences transition uses Board's one cursor: Gray Ogre on board with −3 stop-delta, no whole-transition damage-dealt, and recap hidden.
  await reviewDamage.click();
  await expect(reviewDamage).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);
  await expect(page.locator('.half.opponent [data-testid=stop-delta][data-delta-kind=damage]').first()).toHaveText('−3');
  await expect(page.getByTestId('damage-dealt')).toHaveCount(0);
  await expect(page.getByTestId('recent')).toHaveCount(0);
  // Selecting died inside the same transition shows Gray Ogre gone with its death chip and exact top-bar reading.
  await reviewDeath.click();
  await expect(reviewDeath).toHaveAttribute('aria-current', 'true');
  await expect(reviewDamage).not.toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(0);
  await expect(page.getByTestId('death-chip').first()).toContainText('lethal damage');
  await expect(page.getByTestId('turn-number')).toContainText('Your Lightning Bolt · Gray Ogre died · lethal damage');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await page.screenshot({ path: testInfo.outputPath('review-death-cursor.png') });
  // Clicking the player move's own line from its consequences transition jumps cleanly back to `choice`.
  await reviewCast.click();
  await expect(page.getByTestId('turn-number')).not.toContainText('Esc for now');
  await expect(page.getByTestId('study-open')).toBeVisible();
});

test('normal motion animates a death travelling to the graveyard within 180 ms with its rule', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.routeWebSocket('**/ws/play', socket => {
    const server = socket.connectToServer();
    socket.onMessage(message => {
      if (typeof message !== 'string') { server.send(message); return; }
      const payload = JSON.parse(message);
      if (payload.type === 'new_game') payload.config.seed = 11;
      server.send(JSON.stringify(payload));
    });
  });

  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await selectDeck(page, 'hero', 'interactive');
  await selectDeck(page, 'villain', 'interactive');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();

  expect(await playToDeath(page)).toBe(true);

  const toggles = page.locator('[data-testid=consequence-toggle][aria-expanded="false"]');
  for (let i = 0; i < await toggles.count(); i += 1) {
    await toggles.nth(0).click();
  }

  const damageButton = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /3 damage to Gray Ogre/ }).first();
  await damageButton.click();
  await expect(page.locator('.half.opponent [data-testid=board-card]').filter({ hasText: 'Gray Ogre' })).toHaveCount(1);

  // Step from damage to death: Gray Ogre travels to the opponent's graveyard within 180 ms with `lethal damage`.
  await page.keyboard.press('ArrowRight');
  const travel = page.getByTestId('death-travel').first();
  await expect(travel).toBeVisible();
  await expect(travel).toHaveAttribute('data-death-rule', 'lethal damage');
  await expect(travel).toHaveAttribute('data-death-side', 'opponent');

  const anim = await travel.evaluate(el => {
    const running = el.getAnimations().find(a => a.id === 'death-travel');
    const timing = running?.effect?.getComputedTiming();
    const keyframes = running?.effect instanceof KeyframeEffect ? running.effect.getKeyframes() : [];
    return {
      found: Boolean(running),
      duration: typeof timing?.duration === 'number' ? timing.duration : null,
      toTranslate: String(keyframes.at(-1)?.translate ?? ''),
    };
  });
  expect(anim.found).toBe(true);
  expect(anim.duration).toBeLessThanOrEqual(180);
  // Opponent's graveyard is on the left plate, so the card travels left (negative dx).
  expect(anim.toTranslate).toMatch(/-\d+/);
  await expect(page.getByTestId('death-chip').first()).toContainText('lethal damage');
  await page.screenshot({ path: testInfo.outputPath('normal-death-travel.png') });
});

test('triggered ability on the stack draws its target arrow and names itself in the top bar', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'reduce', colorScheme: 'light' });

  const recovery = structuredClone(boltFixture.recovery);
  recovery.reason = 'initial_connect';
  recovery.frame.match_id = 'ability-arrow-proof';
  recovery.frame.revision = 20;
  recovery.frame.frame_hash = 'ability-arrow-20';
  recovery.frame.projection.turn = {
    turn_number: 6,
    phase: 'COMBAT',
    step: 'COMBAT_DAMAGE_STEP',
    active_player_id: 1,
    agent_player_id: 0,
  };
  recovery.frame.projection.agent.life = 12;
  recovery.frame.projection.opponent.life = 19;
  recovery.frame.projection.agent.stack = [];
  recovery.frame.projection.opponent.stack = [];
  recovery.frame.projection.stack_objects = [
    {
      id: 901,
      kind: 'triggered_ability',
      controller: 1,
      label: 'Vengeful Bloodwitch — Drain',
      source_permanent: 30,
    },
  ];
  recovery.frame.offers = [
    {
      id: 1,
      actor: 0,
      verb: 'pass_priority',
      source: null,
      label: 'Pass priority',
      help: null,
      choices: [],
      confirm_label: 'Pass priority',
      action_type: 'PRIORITY_PASS_PRIORITY',
      focus: [],
    },
  ];
  recovery.frame.action_space = 'PRIORITY';
  recovery.frame.prompt = {
    id: 1,
    actor: 0,
    kind: 'priority',
    title: 'Priority',
    instruction: 'Respond to Vengeful Bloodwitch, or pass priority.',
  };
  const ev = (seq: number, kind: PresentationEvent['kind'], extra: Partial<PresentationEvent> = {}): PresentationEvent => ({
    seq,
    from_revision: 19,
    to_revision: 20,
    caused_by: null,
    group: 1,
    importance: 'normal',
    suggested_ms: 0,
    sound: null,
    kind,
    ...extra,
  });
  recovery.presentation_tail = [
    ev(0, { kind: 'turn_started', player: 1, turn_number: 6 }),
    ev(1, { kind: 'died', objects: [{ entity: 30, incarnation: 1 }] }, {
      rule: 'lethal_damage',
      labels: { objects: { '30:1': 'Vengeful Bloodwitch' } },
    }),
    ev(2, { kind: 'triggered', source: { entity: 30, incarnation: 1 }, controller: 1, stack: 901 }, {
      cause: 1,
      labels: { objects: { '30:1': 'Vengeful Bloodwitch' }, stacks: { '901': 'Vengeful Bloodwitch' } },
    }),
    ev(3, { kind: 'targeted', source: { kind: 'stack', id: 901 }, target: { kind: 'player', id: 0 } }, {
      cause: 2,
      labels: { stacks: { '901': 'Vengeful Bloodwitch' } },
    }),
  ];
  recovery.presentation_cursor = 0;

  const game = gameReply();
  await page.routeWebSocket('**/ws/play', socket => {
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw)) as { type?: string };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation',
          ...game,
          data: recovery.frame.projection,
          actions: [],
          recovery,
        }));
      }
    });
  });

  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toBeVisible();

  // Triggered ability on the stack exposes data-stack-card and draws its target arrow to your plate.
  await expect(page.getByTestId('stack-entry').first()).toHaveAttribute('data-stack-card', '901');
  await expect(page.locator('[data-testid=board-lines] [data-line-kind=target]')).toHaveCount(1);

  // Selecting the trigger in History names the trigger itself (not the parent death) in the top bar.
  const triggerBtn = page.locator('[data-testid=game-log] [data-testid=history-consequence] button[data-history-seq]').filter({ hasText: /Trigger · Vengeful Bloodwitch, targeting you/ });
  await triggerBtn.click();
  await expect(page.getByTestId('turn-number')).toContainText('Turn 6 · Trigger · Vengeful Bloodwitch, targeting you');
  await expect(page.getByTestId('turn-number')).not.toContainText('died');
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await expect(page.getByTestId('recent')).toHaveCount(0);
  await expect(page.locator('[data-testid=board-lines] [data-line-kind=target]')).toHaveCount(1);

  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('ability-target-arrow-wash.png') });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('ability-target-arrow-night.png') });
});

test('a combat with three or more creatures shows the structured combat table on its damage stop', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'reduce', colorScheme: 'light' });

  const recovery = structuredClone(boltFixture.recovery);
  recovery.reason = 'initial_connect';
  recovery.frame.match_id = 'combat-table-proof';
  recovery.frame.revision = 30;
  recovery.frame.frame_hash = 'combat-table-30';
  recovery.frame.projection.turn = {
    turn_number: 6,
    phase: 'COMBAT',
    step: 'COMBAT_END',
    active_player_id: 1,
    agent_player_id: 0,
  };
  recovery.frame.projection.agent.life = 12;
  recovery.frame.projection.opponent.life = 20;
  recovery.frame.projection.agent.stack = [];
  recovery.frame.projection.opponent.stack = [];
  recovery.frame.projection.stack_objects = [];
  recovery.frame.projection.agent.battlefield = [
    {
      id: 20,
      name: 'Beetle-Headed Merchants',
      controller_id: 0,
      tapped: false,
      power: 5,
      toughness: 4,
      damage: 3,
      summoning_sick: false,
      plus1_counters: 0,
      types: { is_creature: true, is_land: false },
    },
  ] as typeof recovery.frame.projection.agent.battlefield;
  recovery.frame.projection.opponent.battlefield = [
    {
      id: 11,
      name: 'Wandering Musicians',
      controller_id: 1,
      tapped: true,
      power: 2,
      toughness: 5,
      damage: 0,
      summoning_sick: false,
      plus1_counters: 0,
      types: { is_creature: true, is_land: false },
    },
  ] as typeof recovery.frame.projection.opponent.battlefield;
  recovery.frame.offers = [
    {
      id: 1,
      actor: 0,
      verb: 'pass_priority',
      source: null,
      label: 'Pass priority',
      help: null,
      choices: [],
      confirm_label: 'Pass priority',
      action_type: 'PRIORITY_PASS_PRIORITY',
      focus: [],
    },
  ];
  recovery.frame.action_space = 'PRIORITY';
  recovery.frame.prompt = {
    id: 1,
    actor: 0,
    kind: 'priority',
    title: 'Priority',
    instruction: 'Pass priority.',
  };
  const ev = (seq: number, kind: PresentationEvent['kind'], extra: Partial<PresentationEvent> = {}): PresentationEvent => ({
    seq,
    from_revision: 29,
    to_revision: 30,
    caused_by: null,
    group: 1,
    importance: 'normal',
    suggested_ms: 0,
    sound: null,
    kind,
    ...extra,
  });
  recovery.presentation_tail = [
    ev(0, { kind: 'turn_started', player: 1, turn_number: 6 }),
    ev(1, { kind: 'attack_group', attackers: [{ entity: 10, incarnation: 1 }, { entity: 11, incarnation: 1 }], defender: { kind: 'player', id: 0 } }, {
      labels: { objects: { '10:1': 'Earth Kingdom Jailer', '11:1': 'Wandering Musicians' } },
    }),
    ev(2, { kind: 'blocked', attacker: { entity: 10, incarnation: 1 }, blockers: [{ entity: 20, incarnation: 1 }] }, {
      labels: { objects: { '10:1': 'Earth Kingdom Jailer', '20:1': 'Beetle-Headed Merchants' } },
    }),
    ev(3, { kind: 'damage', source: { kind: 'object', id: { entity: 10, incarnation: 1 } }, target: { kind: 'object', id: { entity: 20, incarnation: 1 } }, amount: 3 }, {
      labels: { objects: { '10:1': 'Earth Kingdom Jailer', '20:1': 'Beetle-Headed Merchants' } },
    }),
    ev(4, { kind: 'damage', source: { kind: 'object', id: { entity: 20, incarnation: 1 } }, target: { kind: 'object', id: { entity: 10, incarnation: 1 } }, amount: 5 }, {
      labels: { objects: { '20:1': 'Beetle-Headed Merchants', '10:1': 'Earth Kingdom Jailer' } },
    }),
    ev(5, { kind: 'damage', source: { kind: 'object', id: { entity: 11, incarnation: 1 } }, target: { kind: 'player', id: 0 }, amount: 2 }, {
      labels: { objects: { '11:1': 'Wandering Musicians' } },
    }),
    ev(6, { kind: 'died', objects: [{ entity: 10, incarnation: 1 }] }, {
      cause: 4,
      rule: 'lethal_damage',
      labels: { objects: { '10:1': 'Earth Kingdom Jailer' } },
    }),
  ];
  recovery.presentation_cursor = 0;

  const game = gameReply();
  await page.routeWebSocket('**/ws/play', socket => {
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw)) as { type?: string };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation',
          ...game,
          data: recovery.frame.projection,
          actions: [],
          recovery,
        }));
      }
    });
  });

  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toBeVisible();

  // Clicking the combat damage line opens the structured combat table for the 3-creature fight.
  const dmgLine = page.locator('[data-testid=game-log] button[data-history-seq]').filter({ hasText: /Earth Kingdom Jailer dealt 3/ }).first();
  await dmgLine.click();
  const table = page.getByTestId('combat-table');
  await expect(table).toBeVisible();
  await expect(table).toHaveAttribute('data-creatures', '3');
  await expect(table.getByTestId('combat-table-row')).toHaveCount(2);
  await expect(table).toContainText('Earth Kingdom Jailer');
  await expect(table).toContainText('Beetle-Headed Merchants');
  await expect(table).toContainText('Wandering Musicians');
  await expect(table).toContainText('died · lethal damage');
  await expect(table).toContainText('you −2');

  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('combat-table-wash.png') });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForTimeout(150);
  await page.screenshot({ path: testInfo.outputPath('combat-table-night.png') });
});
