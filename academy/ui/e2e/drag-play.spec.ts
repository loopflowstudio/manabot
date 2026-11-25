import { expect, test, type Locator, type Page } from '@playwright/test';

import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { CardState, Command, DecisionObjectView, ExperienceFrame, InteractionOffer, Observation, RecoveryEnvelope, SubjectRef } from '../src/lib/types';
import type { RestoredReplayDecision } from '../src/lib/replay-index';
import type { CombatPlan } from '../src/lib/board/combat';
import { startCombat } from './fixtures/combat';
import { tableSnapshot } from './fixtures/table';

// Dragging on the board: a drop goes where a click goes. These drive the
// real pointer against a controlled authority and read the Commands it got.

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };
const REVISION = 20;
const BOLT = 71;
const SHOCK = 72;
const ISLAND = 73;
const MOUNTAIN = 61;

function land(id: number, name: string): Observation['agent']['battlefield'][number] {
  return {
    id, name, controller_id: 0, tapped: false, damage: 0, summoning_sick: false, power: null, toughness: null,
    base_power: null, base_toughness: null, plus1_counters: 0,
    types: { is_creature: false, is_land: true, is_spell: false, is_artifact: false, is_enchantment: false, is_planeswalker: false, is_battle: false },
  } as Observation['agent']['battlefield'][number];
}
function card(id: number, name: string, isLand = false): CardState {
  return {
    id, registry_key: id, name, zone: 'HAND', owner_id: 0, power: 0, toughness: 0, mana_value: isLand ? 0 : 1,
    types: { is_creature: false, is_land: isLand, is_spell: !isLand, is_artifact: false, is_enchantment: false, is_planeswalker: false, is_battle: false },
  };
}
function offer(id: number, verb: InteractionOffer['verb'], label: string, actionType: string, focus: number[]): InteractionOffer {
  return { id, actor: 0, verb, source: null, label, help: null, choices: [], confirm_label: label, action_type: actionType, focus };
}

/** Your main phase: Lightning Bolt casts one way, Shock two ways, Island plays, Opt does nothing. */
function mainPhase(revision: number, hand: number[] = [BOLT, SHOCK, ISLAND, 74]): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const projection = structuredClone(base.projection);
  projection.turn = { turn_number: 3, phase: 'PRECOMBAT_MAIN', step: 'PRECOMBAT_MAIN_STEP', active_player_id: 0, agent_player_id: 0 };
  const cards = { [BOLT]: card(BOLT, 'Lightning Bolt'), [SHOCK]: card(SHOCK, 'Shock'), [ISLAND]: card(ISLAND, 'Island', true), 74: card(74, 'Opt') };
  projection.agent.hand = hand.map(id => cards[id]);
  projection.agent.battlefield = [land(MOUNTAIN, 'Mountain'), land(62, 'Mountain')];
  projection.opponent.battlefield = [];
  const offers = [
    offer(0, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', []),
    offer(1, 'cast', 'Cast Lightning Bolt', 'PRIORITY_CAST_SPELL', [BOLT]),
    offer(2, 'cast', 'Cast Shock', 'PRIORITY_CAST_SPELL', [SHOCK]),
    offer(3, 'cast', 'Cast Shock with kicker', 'PRIORITY_CAST_SPELL', [SHOCK]),
    offer(4, 'play_land', 'Play Island', 'PRIORITY_PLAY_LAND', [ISLAND]),
  ].filter(candidate => candidate.focus.every(id => hand.includes(id)));
  return {
    ...base, match_id: 'drag-play-proof', revision, frame_hash: `drag-frame-${revision}`,
    prompt: { id: 100 + revision, actor: 0, kind: 'priority', title: 'Your main phase', instruction: 'Play a card or pass' },
    projection, offers, action_space: 'PRIORITY', deck_names: { hero: 'Red', villain: 'Blue' },
  };
}

interface Authority { commands: Command[]; previews: number[] }

/** Accepts every Command but `refuse`'s offers, which it rejects at the same revision. */
async function installAuthority(page: Page, {
  refuse = [], initial = mainPhase(REVISION), after, study, stopEcho, commandReply,
}: { refuse?: number[]; initial?: ExperienceFrame; study?: RestoredReplayDecision; stopEcho?: Promise<void>; commandReply?: Promise<void>; after?: (command: Command) => ExperienceFrame } = {}): Promise<Authority> {
  const authority: Authority = { commands: [], previews: [] };
  let current = initial;
  await page.route('**/api/sessions/*/payment-preview', async route => {
    const request = route.request().postDataJSON() as { offer_id: number } & Record<string, unknown>;
    authority.previews.push(request.offer_id);
    await route.fulfill({ json: { ...request, viewer: 0, preview: {
      state: 'exact', explanation: 'Mountain pays it.', cost: '{R}', pool_after_production: null, pool_after_spending: null,
      sources: [{ source: { kind: 'object', id: { entity: MOUNTAIN, incarnation: 0 } }, label: 'Mountain', output: { amounts: [0, 0, 0, 1, 0, 0], classes: [] } }],
    } } });
  });
  await page.routeWebSocket('**/ws/play', socket => {
    const recovery = (frame: ExperienceFrame): RecoveryEnvelope => ({ ...structuredClone(boltFixture.recovery), reason: 'initial_connect', frame,
      presentation_tail: [], presentation_cursor: 0, accepted_commands: [], replay_cursor: frame.revision });
    const table = tableSnapshot();
    socket.onMessage(async raw => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command; stops?: Pick<ExperienceFrame['stops'], 'my' | 'opponent'> };
      if (message.type === 'new_game' || message.type === 'resume' || message.type === 'set_stops') {
        if (message.type === 'set_stops') {
          await stopEcho;
          current = { ...current, stops: { ...current.stops, ...message.stops } };
        }
        socket.send(JSON.stringify({ type: 'observation', table, data: current.projection, actions: [],
          session_id: 'drag-session', resume_token: 'drag-resume-token', recovery: recovery(current) }));
        if (study) socket.send(JSON.stringify({ type: 'decision_restored', address: study.address, restored: study }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      authority.commands.push(command);
      await commandReply;
      if (refuse.includes(command.offer_id)) {
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: command.command_id, code: 'stale_revision', message: 'That play is no longer available.',
            current_revision: current.revision, current_prompt: current.prompt!.id }, recovery: { ...recovery(current), reason: 'command_rejected' } }));
        return;
      }
      const played = current.offers.find(candidate => candidate.id === command.offer_id)?.focus[0];
      const next = after?.(command) ?? mainPhase(current.revision + 1, current.projection.agent.hand.map(held => held.id).filter(id => id !== played));
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted', update: {
        base_revision: command.expected_revision, frame: next, presentation: [],
        receipt: { command_id: command.command_id, actor: 0, accepted_at: command.expected_revision, resulting_revision: next.revision, resulting_frame_hash: next.frame_hash },
      } }));
      current = next;
    });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
  return authority;
}

const inHand = (page: Page, name: string): Locator => page.locator(`.hand [data-testid=board-card][data-card-name="${name}"]`);
const ghost = (page: Page): Locator => page.getByTestId('drag-ghost');
const planned = (page: Page): Locator => page.locator('.board [data-testid=board-card][data-payment-source]');

async function center(locator: Locator): Promise<{ x: number; y: number }> {
  const box = (await locator.boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}
/** Press a hand card on its uncovered left edge and carry it to `to`, without releasing. */
async function carry(page: Page, from: Locator, to: { x: number; y: number }): Promise<void> {
  // A card that just settled back into its slot is pressed where it rests.
  await expect.poll(() => from.evaluate(element => element.getAnimations().length)).toBe(0);
  const box = (await from.boundingBox())!;
  const start = { x: box.x + 12, y: box.y + box.height / 2 };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x, start.y - 20, { steps: 3 });
  await page.mouse.move(to.x, to.y, { steps: 8 });
}
const field = (page: Page): Promise<{ x: number; y: number }> => center(page.locator('.half.you .rows'));

test('a card dragged onto the battlefield is cast, with its payment shown on the way', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const authority = await installAuthority(page);
  const bolt = inHand(page, 'Lightning Bolt');

  // Rest on the card first, as a person reaching for it would: inspect does not open once it is carried.
  const fieldPoint = await field(page);
  await bolt.hover({ position: { x: 12, y: 60 } });
  await carry(page, bolt, fieldPoint);
  await expect(ghost(page)).toBeVisible();
  await expect(page.getByTestId('drop-zone')).toHaveAttribute('data-over', 'true');
  await expect(page.getByTestId('drop-zone')).toContainText('Cast Lightning Bolt');
  await expect(planned(page)).toHaveCount(1);
  await expect(page.locator(`[data-object-id="${MOUNTAIN}"]`)).toHaveAttribute('data-payment-source', '1');
  await page.waitForTimeout(500);
  await expect(page.getByTestId('card-preview')).toHaveCount(0);
  // The card in hand stays in its slot as the place it returns to.
  await expect(bolt).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('drag-over-field.png') });
  expect(authority.commands).toEqual([]);

  // Over the hand the frame says releasing cancels.
  const boltPoint = await center(bolt);
  await page.mouse.move(boltPoint.x, boltPoint.y, { steps: 4 });
  await expect(page.getByTestId('drop-zone')).toHaveAttribute('data-over', 'false');
  await expect(page.getByTestId('drop-zone')).toContainText('Release to cancel');
  await page.mouse.move(fieldPoint.x, fieldPoint.y, { steps: 4 });
  await page.mouse.up();

  await expect.poll(() => authority.commands.length).toBe(1);
  expect(authority.commands.map(command => [command.expected_revision, command.offer_id])).toEqual([[REVISION, 1]]);
  await expect(bolt).toHaveCount(0);
  await expect(ghost(page)).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('a carried card released off the battlefield or on Escape goes back with nothing sent', async ({ page }) => {
  const authority = await installAuthority(page);
  const bolt = inHand(page, 'Lightning Bolt');

  // Released over the hand: back in its slot, and the release is not a click on it.
  await carry(page, bolt, await center(page.locator('.hand')));
  await page.mouse.up();
  await expect(ghost(page)).toHaveCount(0);
  await expect(bolt).toHaveAttribute('data-travel-id', `card:${BOLT}`);

  // Escape mid-drag ends the drag, even over the battlefield.
  await carry(page, bolt, await field(page));
  await expect(ghost(page)).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(ghost(page)).toHaveCount(0);
  await page.mouse.up();
  await page.waitForTimeout(100);
  expect(authority.commands).toEqual([]);

  // A short press is still a click: the next real click plays the land.
  await inHand(page, 'Island').click({ position: { x: 12, y: 60 } });
  await expect.poll(() => authority.commands.map(command => command.offer_id)).toEqual([4]);
});

test('a card with several plays opens its choices where it was dropped', async ({ page }) => {
  const authority = await installAuthority(page);
  const shock = inHand(page, 'Shock');

  await carry(page, shock, await field(page));
  await expect(page.getByTestId('drop-zone')).toContainText('Choose how to play Shock');
  await page.mouse.up();
  const menu = page.getByTestId('card-menu');
  await expect(menu).toBeVisible();
  await expect(ghost(page)).toHaveAttribute('data-holding', 'menu');
  const dropped = (await field(page)).y;
  expect(Math.abs((await menu.boundingBox())!.y - dropped)).toBeLessThan(260);
  expect(authority.commands).toEqual([]);

  // Cancel: the card goes back and focus returns to it.
  await menu.getByRole('menuitem', { name: /Cancel/ }).click();
  await expect(menu).toHaveCount(0);
  await expect(ghost(page)).toHaveCount(0);
  await expect(shock).toBeFocused();

  // Choosing one commits exactly that offer.
  await carry(page, shock, await field(page));
  await page.mouse.up();
  await menu.getByRole('menuitem', { name: /Cast Shock with kicker/ }).click();
  await expect.poll(() => authority.commands.map(command => [command.expected_revision, command.offer_id])).toEqual([[REVISION, 3]]);
});

test('choosing a different offer returns the carried card while that Command is pending', async ({ page }) => {
  const reply = Promise.withResolvers<void>();
  const authority = await installAuthority(page, { commandReply: reply.promise });
  const shock = inHand(page, 'Shock');
  await carry(page, shock, await field(page));
  await page.mouse.up();
  await expect(ghost(page)).toHaveAttribute('data-holding', 'menu');
  await page.getByTestId('action-option').filter({ hasText: 'Play Island' }).click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'false');
  await expect(ghost(page)).toHaveCount(0);
  await expect(shock).toHaveAttribute('data-travel-id', `card:${SHOCK}`);
  expect(authority.commands.map(c => c.offer_id)).toEqual([4]);
  reply.resolve();
  await expect(inHand(page, 'Island')).toHaveCount(0);
});

test('dropping a hand card over a battlefield tray plays the carried card', async ({ page }) => {
  const initial = mainPhase(REVISION);
  initial.offers.push(offer(5, 'cast', 'Cast from exile', 'PRIORITY_CAST_SPELL', [75]));
  const authority = await installAuthority(page, { initial });
  await carry(page, inHand(page, 'Lightning Bolt'), await center(page.getByTestId('tray')));
  await expect(page.getByTestId('drop-zone')).toContainText('Cast Lightning Bolt');
  await page.mouse.up();
  await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.offer_id])).toEqual([[REVISION, 1]]);
});

test('a refused drop returns the card to hand', async ({ page }) => {
  const authority = await installAuthority(page, { refuse: [1] });
  const bolt = inHand(page, 'Lightning Bolt');
  await carry(page, bolt, await field(page));
  await page.mouse.up();
  await expect.poll(() => authority.commands.length).toBe(1);
  await expect(ghost(page)).toHaveCount(0);
  await expect(bolt).toHaveAttribute('data-travel-id', `card:${BOLT}`);
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
});

const renderId = (entity: number) => ({ entity, incarnation: 0 });
function subject(id: number, label: string, ref: SubjectRef = { kind: 'object', id: renderId(id) }): DecisionObjectView {
  return { entity_id: id, display_id: id, subject: ref, label };
}
const otter = subject(83, 'Otter-Penguin');
const cub = subject(84, 'Badgermole Cub');
const tiger = subject(85, 'Tiger-Seal');
const ally = subject(86, 'Ally');
const ability = subject(90, 'Triggered ability', { kind: 'stack', id: 90 });
const opponent = subject(1, 'Opponent', { kind: 'player', id: 1 });
function creature(object: DecisionObjectView, controller: number, attacking = false): Observation['agent']['battlefield'][number] {
  return { id: object.display_id, name: object.label, controller_id: controller, tapped: false, damage: 0, summoning_sick: false,
    power: 2, toughness: 2, base_power: 2, base_toughness: 2, plus1_counters: 0, attacking };
}
const permanent = (page: Page, id: number) => page.locator(`.rows [data-object-id="${id}"]`);
const stackEntry = (page: Page, id: number) => page.locator(`[data-stack-entry="${id}"]`);

function targets(source: 'spell' | 'trigger'): ExperienceFrame {
  const frame = mainPhase(REVISION);
  frame.action_space = 'CHOOSE_TARGET';
  frame.prompt = { ...frame.prompt!, kind: 'choose_target', title: 'Choose a target', instruction: 'Choose a target' };
  frame.projection.agent.battlefield.push(creature(otter, 0));
  frame.projection.opponent.battlefield = [creature(cub, 1)];
  frame.projection.stack_objects = [{ id: ability.display_id, kind: 'triggered_ability', controller: 1, label: ability.label }];
  frame.projection.decision_context = {
    recent_facts: [], actor: 0, stage: source === 'spell' ? 'spell_targets' : 'trigger_targets', instruction: 'Choose a target',
    source: source === 'spell' ? subject(BOLT, 'Lightning Bolt') : otter, targets: [], reserved_sacrifice: null,
    mana_cost: null, chosen_mode: null, ...(source === 'spell' ? { announcing: 'spell' } : {}),
  };
  frame.offers = [cub, opponent, ability].map((object, i) => ({
    ...offer(10 + i, 'choose', `Target ${object.label}`, 'CHOOSE_TARGET', [object.display_id]), subjects: [{ role: 'target', object }],
  }));
  return frame;
}

for (const [source, target, id] of [
  ['spell', 'creature', 10], ['spell', 'plate', 11], ['spell', 'stack ability', 12], ['trigger', 'creature', 10],
] as const) {
  test(`drag a ${source} to a ${target} to choose its offer`, async ({ page }) => {
    const authority = await installAuthority(page, { initial: targets(source) });
    const from = source === 'spell' ? page.locator('[data-announcing=spell]') : permanent(page, otter.display_id);
    const plate = page.locator('[data-plate-side=opponent]');
    const to = target === 'plate' ? plate.getByTestId('player-target') : target === 'creature' ? permanent(page, cub.display_id) : stackEntry(page, ability.display_id);
    // The whole plate accepts the drop, including the mana pool outside its button.
    await carry(page, from, await center(target === 'plate' ? plate.getByTestId('mana-pool') : to));
    await expect(page.getByTestId('draft-line')).toHaveAttribute('data-line-kind', 'target');
    await expect(page.locator('[data-connect-target]')).toHaveCount(1);
    await expect(permanent(page, cub.display_id)).toHaveAttribute('data-connect-target', 'true');
    await expect(to).toHaveAttribute('data-drop', 'true');
    await expect(planned(page)).toHaveCount(1);
    expect(authority.previews).toContain(id);
    expect(authority.commands).toEqual([]);
    await page.mouse.up();
    await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.offer_id])).toEqual([[REVISION, id]]);
    await expect(page.getByTestId('draft-line')).toHaveCount(0);
  });
}

function blocks(blocker = otter, revision = REVISION): ExperienceFrame {
  const frame = mainPhase(revision, []);
  frame.action_space = 'DECLARE_BLOCKER';
  frame.stops.auto_pass = true;
  frame.prompt = { ...frame.prompt!, kind: 'declare_blocker', title: 'Choose a block', instruction: 'Choose a block' };
  frame.projection.turn = { ...frame.projection.turn, active_player_id: 1, phase: 'COMBAT', step: 'DECLARE_BLOCKERS_STEP' };
  frame.projection.agent.battlefield = [creature(otter, 0), creature(tiger, 0)];
  frame.projection.opponent.battlefield = [creature(cub, 1, true), creature(ally, 1, true), creature(subject(87, 'Flyer'), 1, true)];
  frame.offers = [cub, ally].map((object, i) => ({
    ...offer(20 + i, 'declare_blockers', `Block ${object.label} with ${blocker.label}`, 'DECLARE_BLOCKER', [blocker.display_id, object.display_id]),
    subjects: [{ role: 'choice', object: blocker }, { role: 'target', object }],
  }));
  frame.offers.push({ ...offer(22, 'declare_blockers', 'Do not block', 'NO_BLOCK', [blocker.display_id]), subjects: [{ role: 'choice', object: blocker }] });
  return frame;
}

test('drag the required blocker to an attacker in a one-at-a-time prompt', async ({ page }) => {
  const initial = blocks();
  initial.offers.pop(); // A required block has no decline button to identify the blocker.
  await page.route('**/combat-preview', route => route.fulfill({ status: 503 }));
  const authority = await installAuthority(page, { initial });
  const from = permanent(page, otter.display_id);
  await carry(page, from, await center(permanent(page, tiger.display_id)));
  await expect(page.getByTestId('draft-line')).toHaveAttribute('data-line-kind', 'block');
  await expect(page.locator('[data-drop]')).toHaveCount(0);
  await expect(page.locator('[data-connect-target]')).toHaveCount(2);
  await expect(permanent(page, cub.display_id)).toHaveCSS('outline-style', 'dashed');
  await expect(permanent(page, ally.display_id)).toHaveCSS('outline-style', 'dashed');
  await page.mouse.up();
  expect(authority.commands).toEqual([]);
  await carry(page, from, await center(permanent(page, cub.display_id)));
  await expect(permanent(page, cub.display_id)).toHaveAttribute('data-drop', 'true');
  await page.mouse.up();
  await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.offer_id])).toEqual([[REVISION, 20]]);
});

test('dragging a drafted block reassigns it and Escape restores the click-selected blocker', async ({ page }) => {
  await page.route('**/combat-preview', route => {
    const request = route.request().postDataJSON() as { plan: CombatPlan };
    const blockers = [otter, tiger];
    const steps = blockers.map((object, i) => {
      const assigned = request.plan.blocks.find(b => b.blocker.entity === object.display_id)?.attacker;
      return { revision: REVISION + i, fingerprint: `block-${i}`, creature: renderId(object.display_id),
        action_index: assigned ? (assigned.entity === cub.display_id ? 20 : 21) : 22, attack: false, blocks: assigned ?? null };
    });
    return route.fulfill({ json: { ...request, viewer: 0, preview: { kind: 'blockers', steps, refused: null, declared_attackers: [],
      declared_blocks: request.plan.blocks, declarable: blockers.map(object => ({ creature: renderId(object.display_id), label: object.label,
        can_block: [renderId(cub.display_id), renderId(ally.display_id)], must_block: false, must_attack: false })) } } });
  });
  const stopEcho = Promise.withResolvers<void>();
  const authority = await installAuthority(page, { initial: blocks(), stopEcho: stopEcho.promise, after: command => command.expected_revision === REVISION
    ? blocks(tiger, REVISION + 1) : mainPhase(REVISION + 2, []) });
  await expect(page.getByTestId('combat-draft')).toHaveAttribute('data-kind', 'blockers');
  const from = permanent(page, otter.display_id);
  const other = permanent(page, tiger.display_id);
  const first = permanent(page, cub.display_id);
  const second = permanent(page, ally.display_id);
  await carry(page, from, await center(first));
  await expect(page.locator('[data-connect-target]')).toHaveCount(2);
  await expect(first).toHaveAttribute('data-connect-target', 'true');
  await expect(second).toHaveAttribute('data-connect-target', 'true');
  await expect(first).toHaveAttribute('data-drop', 'true');
  await expect(first).toHaveCSS('outline-style', 'solid');
  await expect(second).toHaveCSS('outline-style', 'dashed');
  await page.mouse.up();
  await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Badgermole Cub');
  await other.click();
  await expect(other).toHaveAttribute('data-draft', 'armed');

  // Changing a priority stop holds input, including Space, while keeping the plan and the armed blocker.
  await page.getByTestId('bar-stop-my-upkeep').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'false');
  await expect(page.getByTestId('primary-action')).toBeDisabled();
  await expect(page.getByTestId('combat-clear')).toBeDisabled();
  await expect(page.getByTestId('per-creature-choices').locator('summary')).toBeVisible();
  await expect(page.getByTestId('action-option').first()).toBeHidden();
  await first.focus();
  await page.keyboard.press('Enter');
  await page.keyboard.press('Space');
  await expect(other).toHaveAttribute('data-draft', 'armed');
  await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Badgermole Cub');
  stopEcho.resolve();
  await expect(page.getByTestId('primary-action')).toBeEnabled();
  await expect(other).toHaveAttribute('data-draft', 'armed');
  expect(authority.commands).toEqual([]);

  await carry(page, from, await center(second));
  await expect(page.getByTestId('draft-line')).toHaveAttribute('data-line-kind', 'block');
  await expect(page.locator('[data-line-kind=block]')).toHaveCount(1);
  await expect(second).toHaveAttribute('data-drop', 'true');
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(page.getByTestId('draft-line')).toHaveCount(0);
  await expect(page.locator('[data-line-kind=block]')).toHaveCount(1);
  await expect(other).toHaveAttribute('data-draft', 'armed');
  await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Badgermole Cub');

  await carry(page, from, await center(second));
  await page.mouse.up();
  await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Ally');
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  expect(authority.commands).toEqual([]);
  await page.getByTestId('primary-action').click();
  await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.offer_id])).toEqual([[REVISION, 21], [REVISION + 1, 22]]);
});

test('a restored Study decision keeps clickable choices without a targeting drag', async ({ page }) => {
  const frame = targets('spell');
  const first = frame.offers[0];
  frame.offers.push({ ...first, id: 18, label: 'Choose the second offer' });
  const study: RestoredReplayDecision = {
    address: 'drag-study', frame, revision: frame.revision, ordinal: 0, viewer: 0, presentation_cursor: 0, continuation: [], offer: first,
    command: { command_id: 'recorded', match_id: frame.match_id, expected_revision: frame.revision, prompt_id: frame.prompt!.id, offer_id: first.id, answers: [] },
  };
  const authority = await installAuthority(page, { initial: frame, study });
  await expect(page.getByText(/Study position\. Choosing an offer/)).toBeVisible();
  const recipient = permanent(page, cub.display_id);
  await carry(page, page.locator('[data-announcing=spell]'), await center(recipient));
  await expect(page.getByTestId('draft-line')).toHaveCount(0);
  await page.mouse.up();
  await expect(page.getByTestId('card-menu')).toHaveCount(0);
  await recipient.click();
  await expect(page.getByTestId('card-menu').getByRole('menuitem', { name: /Choose the second offer/ })).toBeVisible();
  expect(authority.commands).toEqual([]);
});

test('a targeted creature with two offers opens its choices from an asked stack ability', async ({ page }) => {
  const frame = targets('trigger');
  frame.projection.decision_context!.source = ability;
  frame.projection.stack_objects![0].controller = 0;
  frame.offers.push({ ...frame.offers[0], id: 18, label: 'Choose the second offer' });
  const authority = await installAuthority(page, { initial: frame });
  const source = stackEntry(page, ability.display_id);
  const to = permanent(page, cub.display_id);
  await carry(page, source, await center(to));
  await expect(to).toHaveAttribute('data-drop', 'true');
  await expect(planned(page)).toHaveCount(0);
  await page.mouse.up();
  const menu = page.getByTestId('card-menu');
  await expect(menu).toBeVisible();
  expect(authority.commands).toEqual([]);
  await menu.getByRole('menuitem', { name: /Choose the second offer/ }).click();
  await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.offer_id])).toEqual([[REVISION, 18]]);
});

for (const colorScheme of ['light', 'dark'] as const) {
  test(`attack marquee highlights the current sweep and confirms its draft in ${colorScheme}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.emulateMedia({ colorScheme, reducedMotion: colorScheme === 'light' ? 'no-preference' : 'reduce' });
    const authority = await startCombat(page);
    const otterCard = permanent(page, otter.display_id);
    const tigerCard = permanent(page, tiger.display_id);
    const start = await center(otterCard);
    const end = await center(tigerCard);
    const marquee = page.getByTestId('attack-marquee');
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(end.x, end.y + 20, { steps: 6 });
    await expect(marquee).toBeVisible();
    await expect(otterCard).toHaveAttribute('data-drop', 'true');
    await expect(tigerCard).toHaveAttribute('data-drop', 'true');
    await expect(tigerCard).toHaveAttribute('data-draft', 'can');
    await page.screenshot({ path: testInfo.outputPath(`marquee-${colorScheme}.png`) });
    await page.mouse.move(start.x, start.y + 12);
    await expect(tigerCard).not.toHaveAttribute('data-drop');
    await page.mouse.move(end.x, end.y + 20);
    await page.mouse.up();
    await expect(marquee).toHaveCount(0);
    await expect(otterCard).toHaveAttribute('data-draft', 'in');
    await expect(tigerCard).toHaveAttribute('data-draft', 'in');
    const confirm = page.getByTestId('primary-action');
    await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.commands).toEqual([]);
    await confirm.click();
    await expect.poll(() => authority.commands.map(c => [c.expected_revision, c.prompt_id, c.offer_id]))
      .toEqual([[10, 110, 101], [11, 111, 103]]);
    await expect(page.getByTestId('decision-prompt')).toContainText('No blocks · Drag a blocker onto an attacker, or click each');
  });
}

test('attack marquee starts only in the rows and Escape preserves the draft', async ({ page }) => {
  const authority = await startCombat(page);
  const otterCard = permanent(page, otter.display_id);
  const tigerCard = permanent(page, tiger.display_id);
  await otterCard.click();
  const end = await center(tigerCard);
  const marquee = page.getByTestId('attack-marquee');
  for (const outside of [page.locator('.half.you .plate'), page.locator('.hand')]) {
    const from = await center(outside);
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(end.x, end.y, { steps: 4 });
    await expect(marquee).toHaveCount(0);
    await page.mouse.up();
  }

  // Empty space on the opponent's side can start a sweep across the table.
  const from = await center(page.locator('.half.opponent .rows'));
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 4 });
  await expect(marquee).toBeVisible();
  await expect(tigerCard).toHaveAttribute('data-drop', 'true');
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(marquee).toHaveCount(0);
  await expect(otterCard).toHaveAttribute('data-draft', 'in');
  await expect(tigerCard).toHaveAttribute('data-draft', 'can');
  expect(authority.commands).toEqual([]);
});

test('an attack marquee can draft one exposed copy in a pile', async ({ page }) => {
  const authority = await startCombat(page, { moreTigers: 2 });
  const pile = page.getByTestId('pile-count');
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 0 of 3 attacking');
  const copies = page.locator('.rows [data-card-name="Tiger-Seal"]');
  const boxes = await copies.evaluateAll(cards => cards.map(card => card.getBoundingClientRect().toJSON()));
  const last = boxes.sort((a, b) => b.right - a.right)[0];
  const x = last.right - 2;
  const y = last.top + last.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x, y + 16, { steps: 4 });
  await expect(page.locator('.rows [data-drop]')).toHaveCount(1);
  await page.mouse.up();
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 1 of 3 attacking');
  expect(authority.commands).toEqual([]);
});

test('an attack marquee catches only the visible part of a scrolling row', async ({ page }) => {
  const authority = await startCombat(page);
  // Constrain a two-creature row to one card so its other attacker is scrolled out of view.
  await page.addStyleTag({ content: '.you .creatures .field[role=group] { width: 50px; height: 70px; }' });
  const row = page.getByRole('group', { name: 'Your creatures', exact: true });
  await expect(row).toHaveClass(/overflow/);
  const box = (await row.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + 8, y);
  await page.mouse.down();
  await page.mouse.move(box.x + 140, y + 8, { steps: 4 });
  await expect(row.locator('[data-drop]')).toHaveCount(1);
  await page.mouse.up();
  await expect(row.locator('[data-draft=in]')).toHaveCount(1);
  expect(authority.commands).toEqual([]);
});
