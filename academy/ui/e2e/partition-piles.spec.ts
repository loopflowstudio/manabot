import { expect, test, type Page } from '@playwright/test';

import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { DecisionGroupCommand, DecisionGroupPlan, LiveDecisionGroup, PartitionPilesDecisionGroup } from '../src/lib/decision-group';
import { gameReply } from './fixtures/game';
import type { CardState, Command, ExperienceFrame, FrameUpdate, InteractionOffer, RecoveryEnvelope } from '../src/lib/types';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };
const rid = (entity: number) => ({ entity, incarnation: 0 });

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

function partitionFrame(cursor = 0): ExperienceFrame {
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
        pending: [`inspected: ${CARDS.map(c => c.name).join(', ')}.`],
      },
    },
    offers: [
      offer(0, current.id, current.name, 'face-up'),
      offer(1, current.id, current.name, 'face-down'),
    ],
    deck_names: { hero: 'Foundations Blue', villain: 'Opponent' },
  };
}

function partitionGroup(frame: ExperienceFrame, committedUp = 0, committedDown = 0): PartitionPilesDecisionGroup {
  const offset = committedUp + committedDown;
  return {
    match_id: frame.match_id,
    revision: frame.revision,
    prompt_id: frame.prompt!.id,
    frame_hash: frame.frame_hash,
    kind: 'partition_piles',
    items: CARDS.slice(offset).map(c => ({ item: rid(c.id), label: c.name })),
    committed_face_up: CARDS.slice(0, committedUp).map(c => ({ item: rid(c.id), label: c.name })),
    committed_face_down: CARDS.slice(committedUp, offset).map(c => ({ item: rid(c.id), label: c.name })),
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

function acceptedUpdate(
  command: { command_id: string; expected_revision: number },
  frame: ExperienceFrame,
): FrameUpdate {
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

interface PartitionOptions {
  rejectFirstGroup?: boolean;
  /** Keep the fixture's earlier presentation events so the history can be walked. */
  withHistory?: boolean;
}

interface PartitionAuthority {
  commands: Command[];
  groupCommands: DecisionGroupCommand[];
  previews: DecisionGroupPlan[];
}

async function installPartitionAuthority(
  page: Page,
  options: PartitionOptions = {},
): Promise<PartitionAuthority> {
  const authority: PartitionAuthority = { commands: [], groupCommands: [], previews: [] };
  let current = partitionFrame(0);
  const game = gameReply();
  let currentGroup: LiveDecisionGroup | null = partitionGroup(current);
  let rejectNext = options.rejectFirstGroup ?? false;

  await page.route('**/api/games/*/decision-group-preview', async (route) => {
    const request = route.request().postDataJSON() as {
      match_id: string; revision: number; prompt_id: number; frame_hash: string; plan: DecisionGroupPlan;
    };
    authority.previews.push(request.plan);
    const faceUp = request.plan.kind === 'partition_piles' ? request.plan.face_up : [];
    const upIds = new Set(faceUp.map(item => item.entity));
    const faceDown = currentGroup?.kind === 'partition_piles'
      ? currentGroup.items.filter(c => !upIds.has(c.item.entity)).map(c => c.item)
      : [];
    await route.fulfill({
      json: {
        match_id: request.match_id,
        revision: request.revision,
        prompt_id: request.prompt_id,
        frame_hash: request.frame_hash,
        viewer: 0,
        complete: true,
        refused: null,
        outcome: { kind: 'partition_piles', face_up: faceUp, face_down: faceDown },
      },
    });
  });
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command | DecisionGroupCommand };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation',
          ...game,
          data: current.projection,
          actions: [],
          decision_group: currentGroup,
          recovery: {
            ...structuredClone(boltFixture.recovery),
            reason: 'initial_connect',
            frame: current,
            presentation_tail: options.withHistory ? boltFixture.recovery.presentation_tail : [],
            presentation_cursor: options.withHistory ? boltFixture.recovery.presentation_cursor : 0,
            accepted_commands: [],
            replay_cursor: current.revision,
          },
        }));
        return;
      }
      if (message.type === 'decision_group_command' && message.command) {
        const command = message.command as DecisionGroupCommand;
        authority.groupCommands.push(command);
        if (rejectNext) {
          rejectNext = false;
          socket.send(JSON.stringify({
            type: 'command_outcome',
            status: 'rejected',
            decision_group: currentGroup,
            rejection: {
              command_id: command.command_id,
              code: 'decision_group_refused',
              message: 'Curator split was refused by the authority.',
              current_revision: current.revision,
              refused: {
                item: rid(4),
                label: 'Gray Ogre',
                reason: 'Curator split was refused by the authority.',
              },
            },
          }));
          return;
        }
        current = priorityFrame(command.expected_revision + 5);
        currentGroup = null;
        socket.send(JSON.stringify({
          type: 'command_outcome',
          status: 'accepted',
          decision_group: null,
          update: acceptedUpdate(command, current),
        }));
        return;
      }
      if (message.type === 'command' && message.command) {
        const command = message.command as Command;
        authority.commands.push(command);
        current = partitionFrame(1);
        currentGroup = partitionGroup(current, command.offer_id === 0 ? 1 : 0, command.offer_id === 0 ? 0 : 1);
        socket.send(JSON.stringify({
          type: 'command_outcome',
          status: 'accepted',
          decision_group: currentGroup,
          update: acceptedUpdate(command, current),
        }));
      }
    });
  });
  return authority;
}

async function startPartition(
  page: Page,
  options: PartitionOptions = {},
): Promise<PartitionAuthority> {
  const authority = await installPartitionAuthority(page, options);
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
  await expect(page.getByTestId('partition-panel')).toBeVisible();
  return authority;
}

const partitionCard = (page: Page, id: number) =>
  page.locator(`[data-testid=partition-card][data-object-id="${id}"] [data-testid=board-card]`);

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`Curator pile partition is drafted holistically by ${mode} and submitted in one atomic decision_group_command`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const authority = await startPartition(page);

    // All 5 cards start in face-down pile, previewed via /decision-group-preview, with 0 commands sent.
    await expect(page.getByTestId('partition-panel')).toHaveAttribute('data-group-kind', 'partition_piles');
    await expect(page.locator('[data-testid=partition-pile-up] [data-testid=partition-card]')).toHaveCount(0);
    await expect(page.locator('[data-testid=partition-pile-down] [data-testid=partition-card]')).toHaveCount(5);
    await expect(page.getByTestId('primary-action')).toHaveText(/Confirm split \(0 up, 5 down\)/);
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');

    // Move Gray Ogre (4) and Forest (3) into the face-up pile.
    for (const id of [4, 3]) {
      if (mode === 'pointer') await partitionCard(page, id).click();
      else { await partitionCard(page, id).focus(); await page.keyboard.press('Enter'); }
    }
    await expect(page.locator('[data-testid=partition-pile-up] [data-testid=partition-card]')).toHaveCount(2);
    await expect(page.locator('[data-testid=partition-pile-down] [data-testid=partition-card]')).toHaveCount(3);
    await expect(page.getByTestId('decision-prompt')).toHaveText('2 face-up (Gray Ogre, Forest) · 3 face-down');
    await expect(page.getByTestId('primary-action')).toHaveText(/Confirm split \(2 up, 3 down\)/);
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.commands).toEqual([]);
    expect(authority.groupCommands).toEqual([]);

    // Confirm sends one atomic decision_group_command instead of a 5-step browser send loop.
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.groupCommands.length).toBe(1);
    expect(authority.commands).toEqual([]);
    expect(authority.groupCommands[0]).toMatchObject({
      match_id: 'curator-partition-proof',
      expected_revision: 20,
      prompt_id: 220,
      frame_hash: 'curator-frame-20',
      plan: { kind: 'partition_piles', face_up: [rid(4), rid(3)] },
    });
    await expect(page.getByTestId('partition-panel')).toHaveCount(0);
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
    expect(errors).toEqual([]);
  });
}

test('group rejection banner and inline refusal clear together on draft edit, and raw disclosure stays reachable', async ({ page }) => {
  const authority = await startPartition(page, { rejectFirstGroup: true });
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');

  // Confirm triggers a simulated decision_group_refused rejection -> both rail notice and inline refusal show.
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('partition-refusal')).toContainText('Curator split was refused by the authority.');
  await expect(page.locator('.notices .notice.error')).toContainText('Curator split was refused by the authority.');

  // Editing the draft clears BOTH the inline refusal and the rail error banner.
  await partitionCard(page, 4).click();
  await expect(page.getByTestId('partition-refusal')).toHaveCount(0);
  await expect(page.locator('.notices .notice.error')).toHaveCount(0);
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');

  // Raw per-card disclosure remains reachable and commits a single raw step showing committed state.
  const raw = page.getByTestId('per-creature-choices');
  await expect(raw).toHaveAttribute('data-group-kind', 'partition_piles');
  await raw.locator('summary').click();
  await raw.getByRole('button', { name: 'Put Gray Ogre in the face-up pile' }).click();
  await expect.poll(() => authority.commands.length).toBe(1);
  expect(authority.commands[0].offer_id).toBe(0);
  await expect(page.getByTestId('partition-committed')).toContainText('Face-up: Gray Ogre');
  await expect(page.locator('[data-testid=partition-card]')).toHaveCount(4);
});

test('looking back hides the pile split and holds its Confirm until the board returns to now', async ({ page }) => {
  const authority = await startPartition(page, { withHistory: true });
  const confirm = page.getByTestId('primary-action');
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');

  await page.getByRole('button', { name: /^Stop 1 of 1/ }).click();
  await expect(page.getByTestId('turn-number')).toContainText('Esc for now');
  await expect(page.getByTestId('partition-panel')).toHaveCount(0);
  await expect(confirm).toBeDisabled();
  await expect(confirm).toHaveAttribute('data-draft-ready', 'false');

  await page.keyboard.press('Escape');
  await expect(page.getByTestId('partition-panel')).toBeVisible();
  await expect(confirm).toBeEnabled();
  expect(authority.groupCommands).toHaveLength(0);
});
