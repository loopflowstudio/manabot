import { expect, test, type Page } from '@playwright/test';

import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import { gameReply } from './fixtures/game';
import type { CardState, Command, ExperienceFrame, FrameUpdate, InteractionOffer, RecoveryEnvelope } from '../src/lib/types';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

function card(id: number, name: string): CardState {
  const creature = name === 'Gray Ogre' || name === 'Hill Giant';
  return {
    id,
    owner_id: 0,
    zone: 'LIBRARY',
    registry_key: id,
    name,
    mana_value: creature ? 3 : 0,
    types: {
      is_creature: creature,
      is_land: !creature,
      is_spell: false,
      is_artifact: false,
      is_enchantment: false,
      is_planeswalker: false,
      is_battle: false,
    },
    power: name === 'Gray Ogre' ? 2 : name === 'Hill Giant' ? 3 : 0,
    toughness: name === 'Gray Ogre' ? 2 : name === 'Hill Giant' ? 3 : 0,
  };
}

const CARDS = [
  card(4, 'Gray Ogre'),
  card(5, 'Hill Giant'),
  card(3, 'Forest'),
  card(13, 'Plains'),
  card(14, 'Swamp'),
];

function offer(id: number, cardId: number, name: string, pile: 'face-up' | 'face-down'): InteractionOffer {
  const label = `Put ${name} in the ${pile} pile`;
  return {
    id,
    actor: 0,
    verb: 'choose',
    source: null,
    label,
    help: null,
    choices: [],
    confirm_label: label,
    action_type: 'CHOOSE_MODE',
    focus: [cardId],
  };
}

function partitionFrame(cursor: number, faceUpNames: string[] = []): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const current = CARDS[cursor];
  const revision = 20 + cursor;
  return {
    ...base,
    match_id: 'curator-partition-proof',
    revision,
    frame_hash: `curator-frame-${revision}`,
    action_space: 'MODAL',
    prompt: {
      id: 200 + revision,
      actor: 0,
      kind: 'modal',
      title: 'Put the indicated card into the face-up or face-down pile',
      instruction: 'Put the indicated card into the face-up or face-down pile.',
    },
    projection: {
      ...base.projection,
      agent: { ...base.projection.agent, inspected: [...CARDS] },
      decision_context: {
        actor: 0,
        stage: 'partition_piles',
        instruction: 'Put the indicated card into the face-up or face-down pile. Your opponent sees the face-up cards once the piles are complete.',
        source: {
          entity_id: 0,
          subject: { kind: 'object', id: { entity: 42, incarnation: 2 } },
          display_id: 42,
          label: 'Curator of Destinies',
        },
        targets: [],
        reserved_sacrifice: null,
        mana_cost: null,
        chosen_mode: null,
        recent_facts: [],
        pending: [
          `inspected: ${CARDS.map(c => c.name).join(', ')}.`,
          ...(faceUpNames.length ? [`face-up pile: ${faceUpNames.join(', ')}.`] : []),
        ],
      },
    },
    offers: [
      offer(0, current.id, current.name, 'face-up'),
      offer(1, current.id, current.name, 'face-down'),
    ],
    deck_names: { hero: 'Foundations Blue', villain: 'Opponent' },
  };
}

function priorityFrame(revision: number): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  return {
    ...base,
    match_id: 'curator-partition-proof',
    revision,
    frame_hash: `curator-priority-${revision}`,
    action_space: 'PRIORITY',
    prompt: { id: 200 + revision, actor: 0, kind: 'priority', title: 'Priority', instruction: 'Pass priority' },
    projection: {
      ...base.projection,
      agent: { ...base.projection.agent, inspected: [] },
      decision_context: null,
    },
    offers: [{
      id: 0,
      actor: 0,
      verb: 'pass_priority',
      source: null,
      label: 'Pass priority',
      help: null,
      choices: [],
      confirm_label: 'Pass priority',
      action_type: 'PRIORITY_PASS_PRIORITY',
      focus: [],
    }],
    deck_names: { hero: 'Foundations Blue', villain: 'Opponent' },
  };
}

function acceptedUpdate(command: Command, frame: ExperienceFrame): FrameUpdate {
  return {
    base_revision: command.expected_revision,
    frame,
    presentation: [],
    receipt: {
      command_id: command.command_id,
      actor: 0,
      accepted_at: command.expected_revision,
      resulting_revision: frame.revision,
      resulting_frame_hash: frame.frame_hash,
    },
  };
}

async function installPartitionAuthority(page: Page, interruptAfter?: number): Promise<{ commands: Command[] }> {
  const authority = { commands: [] as Command[] };
  let cursor = 0;
  const faceUp: string[] = [];
  let current = partitionFrame(0);
  const game = gameReply();
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation',
          ...game,
          data: current.projection,
          actions: [],
          recovery: {
            ...structuredClone(boltFixture.recovery),
            reason: 'initial_connect',
            frame: current,
            presentation_tail: [],
            presentation_cursor: 0,
            accepted_commands: [],
            replay_cursor: current.revision,
          },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      authority.commands.push(command);
      if (command.offer_id === 0) faceUp.push(CARDS[cursor].name);
      cursor += 1;
      const next = (interruptAfter !== undefined && authority.commands.length === interruptAfter) || cursor >= CARDS.length
        ? priorityFrame(command.expected_revision + 1)
        : partitionFrame(cursor, faceUp);
      current = next;
      socket.send(JSON.stringify({
        type: 'command_outcome',
        status: 'accepted',
        update: acceptedUpdate(command, next),
      }));
    });
  });
  return authority;
}

async function startPartition(page: Page, interruptAfter?: number): Promise<{ commands: Command[] }> {
  const authority = await installPartitionAuthority(page, interruptAfter);
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
  await expect(page.getByTestId('partition-panel')).toBeVisible();
  return authority;
}

const partitionCard = (page: Page, id: number) =>
  page.locator(`[data-testid=partition-card][data-object-id="${id}"] [data-testid=board-card]`);

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`Curator pile partition is drafted holistically by ${mode} and sent one card at a time on confirm`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const authority = await startPartition(page);

    // All 5 cards are shown together in the face-down pile initially, with 0 commands sent.
    await expect(page.locator('[data-testid=partition-pile-up] [data-testid=partition-card]')).toHaveCount(0);
    await expect(page.locator('[data-testid=partition-pile-down] [data-testid=partition-card]')).toHaveCount(5);
    await expect(page.getByTestId('primary-action')).toHaveText(/Confirm split \(0 up, 5 down\)/);

    // Move Gray Ogre (4) and Forest (3) into the face-up pile.
    for (const id of [4, 3]) {
      if (mode === 'pointer') await partitionCard(page, id).click();
      else { await partitionCard(page, id).focus(); await page.keyboard.press('Enter'); }
    }
    await expect(page.locator('[data-testid=partition-pile-up] [data-testid=partition-card]')).toHaveCount(2);
    await expect(page.locator('[data-testid=partition-pile-down] [data-testid=partition-card]')).toHaveCount(3);
    await expect(page.getByTestId('decision-prompt')).toHaveText('2 face-up (Gray Ogre, Forest) · 3 face-down');
    await expect(page.getByTestId('primary-action')).toHaveText(/Confirm split \(2 up, 3 down\)/);
    expect(authority.commands).toEqual([]);

    // Confirm sends the 5 exact per-card offers in engine order: [up, down, up, down, down].
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.commands.length).toBe(5);
    expect(authority.commands.map(c => [c.expected_revision, c.prompt_id, c.offer_id])).toEqual([
      [20, 220, 0],
      [21, 221, 1],
      [22, 222, 0],
      [23, 223, 1],
      [24, 224, 1],
    ]);
    await expect(page.getByTestId('partition-panel')).toHaveCount(0);
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
    expect(errors).toEqual([]);
  });
}

test('an unexpected frame mid-partition stops the chunked submission cleanly', async ({ page }) => {
  const authority = await startPartition(page, 2);
  await partitionCard(page, 4).click();
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
  await page.waitForTimeout(250);
  expect(authority.commands.map(c => c.offer_id)).toEqual([0, 1]);
  await expect(page.getByTestId('combat-stopped')).toHaveText(/Stopped after 2 of 5: the game asked something unexpected/);
  await expect(page.getByTestId('partition-panel')).toHaveCount(0);
});
