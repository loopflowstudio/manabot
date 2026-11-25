import { expect, test, type Locator, type Page } from '@playwright/test';

import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { CardState, Command, ExperienceFrame, InteractionOffer, Observation, RecoveryEnvelope } from '../src/lib/types';
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
async function installAuthority(page: Page, refuse: number[] = []): Promise<Authority> {
  const authority: Authority = { commands: [], previews: [] };
  let current = mainPhase(REVISION);
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
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({ type: 'observation', table: tableSnapshot(), data: current.projection, actions: [],
          session_id: 'drag-session', resume_token: 'drag-resume-token', recovery: recovery(current) }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      authority.commands.push(command);
      if (refuse.includes(command.offer_id)) {
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: command.command_id, code: 'stale_revision', message: 'That play is no longer available.',
            current_revision: current.revision, current_prompt: current.prompt!.id }, recovery: { ...recovery(current), reason: 'command_rejected' } }));
        return;
      }
      const played = current.offers.find(candidate => candidate.id === command.offer_id)?.focus[0];
      const next = mainPhase(current.revision + 1, current.projection.agent.hand.map(held => held.id).filter(id => id !== played));
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

test('a card dragged onto the battlefield is cast, with its payment shown on the way', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const authority = await installAuthority(page);
  const bolt = inHand(page, 'Lightning Bolt');

  // Rest on the card first, as a person reaching for it would: inspect does not open once it is carried.
  await bolt.hover({ position: { x: 12, y: 60 } });
  await carry(page, bolt, await field(page));
  await expect(ghost(page)).toBeVisible();
  await expect(page.getByTestId('drop-zone')).toHaveAttribute('data-over', 'true');
  await expect(page.getByTestId('drop-zone')).toContainText('Cast Lightning Bolt');
  await expect(planned(page)).toHaveCount(1);
  await expect(page.locator(`[data-object-id="${MOUNTAIN}"]`)).toHaveAttribute('data-payment-source', '1');
  await page.waitForTimeout(500);
  await expect(page.getByTestId('card-preview')).toHaveCount(0);
  // The card in hand stays in its slot as the place it returns to.
  await expect(bolt).toBeVisible();
  expect(authority.commands).toEqual([]);

  // Over the hand the frame says releasing cancels.
  await page.mouse.move((await center(bolt)).x, (await center(bolt)).y, { steps: 4 });
  await expect(page.getByTestId('drop-zone')).toHaveAttribute('data-over', 'false');
  await expect(page.getByTestId('drop-zone')).toContainText('Release to cancel');
  await page.mouse.move((await field(page)).x, (await field(page)).y, { steps: 4 });
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
  await carry(page, bolt, { x: (await center(page.locator('.hand'))).x, y: (await center(page.locator('.hand'))).y });
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

test('a refused drop returns the card to hand', async ({ page }) => {
  const authority = await installAuthority(page, [1]);
  const bolt = inHand(page, 'Lightning Bolt');
  await carry(page, bolt, await field(page));
  await page.mouse.up();
  await expect.poll(() => authority.commands.length).toBe(1);
  await expect(ghost(page)).toHaveCount(0);
  await expect(bolt).toHaveAttribute('data-travel-id', `card:${BOLT}`);
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
});
