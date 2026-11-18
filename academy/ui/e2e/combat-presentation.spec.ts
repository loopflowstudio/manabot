import { tableSnapshot } from './fixtures/table';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

import combatSequence from '../src/lib/fixtures/curated-combat-to-turn.json' with { type: 'json' };
import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  Command,
  ExperienceFrame,
  FrameUpdate,
  InteractionOffer,
  Observation,
  PresentationEvent,
  RecoveryEnvelope,
} from '../src/lib/types';


const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

function permanent(id: number, name: string, controller: number, power: number, toughness: number): Observation['agent']['battlefield'][number] {
  return {
    id, name, controller_id: controller, tapped: false, damage: 0, summoning_sick: false,
    power, toughness, base_power: power, base_toughness: toughness, plus1_counters: 0,
  };
}

/** You: Otter-Penguin (83) and Tiger-Seal (85). The opponent: Badgermole Cub (84). */
function combatObservation(): Observation {
  const observation = structuredClone(boltFixture.recovery.frame.projection);
  observation.turn = {
    turn_number: 4,
    phase: 'COMBAT',
    step: 'DECLARE_ATTACKERS_STEP',
    active_player_id: 0,
    agent_player_id: 0,
  };
  observation.agent.battlefield = [permanent(83, 'Otter-Penguin', 0, 2, 2), permanent(85, 'Tiger-Seal', 0, 3, 3)];
  observation.opponent.battlefield = [permanent(84, 'Badgermole Cub', 1, 1, 1)];
  return observation;
}

function subject(id: number, label: string): NonNullable<InteractionOffer['subjects']>[number]['object'] {
  return { entity_id: id, subject: { kind: 'object', id: { entity: id, incarnation: 0 } }, display_id: id, label };
}

/** A combat offer as the authority builds it: the attack names its creature as source, a block names the attacker as target. */
function offer(
  id: number,
  verb: 'declare_attackers' | 'declare_blockers' | 'pass_priority',
  label: string,
  actionType: string,
  focus: number[],
  subjects: InteractionOffer['subjects'] = undefined,
): InteractionOffer {
  return { id, actor: 0, verb, source: null, label, help: null, choices: [], confirm_label: label, action_type: actionType, focus, ...(subjects ? { subjects } : {}) };
}

const OTTER = subject(83, 'Otter-Penguin');
const CUB = subject(84, 'Badgermole Cub');
const TIGER = subject(85, 'Tiger-Seal');

type Stage = 'otter' | 'tiger' | 'block' | 'priority';
const REVISION: Record<Stage, number> = { otter: 10, tiger: 11, block: 12, priority: 13 };

/** The engine asks one creature at a time: Otter-Penguin, then Tiger-Seal; then the Cub attacks and Otter-Penguin may block. */
function frameAt(stage: Stage, moreTigers = 0): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const revision = REVISION[stage];
  const observation = combatObservation();
  for (let i = 0; i < moreTigers; i++) observation.agent.battlefield.push(permanent(86 + i, 'Tiger-Seal', 0, 3, 3));
  const actionSpace = stage === 'block' ? 'DECLARE_BLOCKER' : stage === 'priority' ? 'PRIORITY' : 'DECLARE_ATTACKER';
  if (stage === 'tiger') observation.agent.battlefield[0] = { ...observation.agent.battlefield[0], attacking: true, tapped: true };
  if (stage === 'block') {
    observation.turn = { ...observation.turn, active_player_id: 1, step: 'DECLARE_BLOCKERS_STEP' };
    observation.opponent.battlefield[0].attacking = true;
  }
  if (stage === 'priority') observation.turn = { ...observation.turn, active_player_id: 1, phase: 'COMBAT', step: 'COMBAT_END' };
  const offers = {
    otter: [
      offer(101, 'declare_attackers', 'Attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'source', object: OTTER }]),
      offer(102, 'declare_attackers', 'Do not attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'choice', object: OTTER }]),
    ],
    tiger: [
      offer(103, 'declare_attackers', 'Attack with Tiger-Seal', 'DECLARE_ATTACKER', [85], [{ role: 'source', object: TIGER }]),
      offer(104, 'declare_attackers', 'Do not attack with Tiger-Seal', 'DECLARE_ATTACKER', [85], [{ role: 'choice', object: TIGER }]),
    ],
    block: [
      offer(201, 'declare_blockers', 'Block Badgermole Cub with Otter-Penguin', 'DECLARE_BLOCKER', [83, 84],
        [{ role: 'choice', object: OTTER }, { role: 'target', object: CUB }]),
      offer(202, 'declare_blockers', 'Otter-Penguin: do not block', 'DECLARE_BLOCKER', [83], [{ role: 'choice', object: OTTER }]),
    ],
    priority: [offer(301, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', [])],
  }[stage];
  return {
    ...base,
    match_id: 'combat-prompt-proof',
    revision,
    frame_hash: `combat-frame-${revision}`,
    prompt: { id: 100 + revision, actor: 0, kind: actionSpace.toLowerCase(), title: 'Choose a combat action', instruction: 'Choose a combat action' },
    projection: observation,
    offers,
    action_space: actionSpace,
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}

/** What the authority reports after the block: the block, damage both ways, nobody dies. */
function combatEvents(from: number): PresentationEvent[] {
  const event = (seq: number, kind: PresentationEvent['kind']): PresentationEvent => ({
    seq, from_revision: from, to_revision: from + 1, caused_by: null, group: 0, importance: 'normal', suggested_ms: 80, sound: null, kind,
  });
  return [
    event(0, { kind: 'blocked', attacker: { entity: 84, incarnation: 0 }, blockers: [{ entity: 83, incarnation: 0 }] }),
    event(1, { kind: 'damage', source: { kind: 'object', id: { entity: 84, incarnation: 0 } }, target: { kind: 'object', id: { entity: 83, incarnation: 0 } }, amount: 1 }),
    event(2, { kind: 'damage', source: { kind: 'object', id: { entity: 83, incarnation: 0 } }, target: { kind: 'object', id: { entity: 84, incarnation: 0 } }, amount: 2 }),
  ];
}

function acceptedUpdate(command: Command, frame: ExperienceFrame, presentation: PresentationEvent[] = []): FrameUpdate {
  return {
    base_revision: command.expected_revision,
    frame,
    presentation,
    receipt: {
      command_id: command.command_id, actor: 0, accepted_at: command.expected_revision,
      resulting_revision: frame.revision, resulting_frame_hash: frame.frame_hash,
    },
  };
}

interface Authority {
  commands: Command[];
  previews: { revision: number; plan: { attackers: { entity: number }[]; blocks: { blocker: { entity: number }; attacker: { entity: number } }[] } }[];
}

const id = (entity: number) => ({ entity, incarnation: 0 });

/**
 * A combat authority: each Command answers the frame it was made at, and the
 * combat preview answers a drafted plan with the per-creature offers the
 * engine would ask for. `after` replaces the frame that follows a revision,
 * to stand for a decision the preview did not predict.
 */
async function installCombatAuthority(page: Page, after: Partial<Record<number, Stage>> = {}, moreTigers = 0): Promise<Authority> {
  const authority: Authority = { commands: [], previews: [] };
  let current = frameAt('otter', moreTigers);
  await page.route('**/api/sessions/*/combat-preview', async route => {
    const request = route.request().postDataJSON() as { match_id: string; revision: number; prompt_id: number; frame_hash: string; plan: Authority['previews'][number]['plan'] };
    authority.previews.push({ revision: request.revision, plan: request.plan });
    const attacks = new Set(request.plan.attackers.map(attacker => attacker.entity));
    const step = (revision: number, creature: number, actionIndex: number, attack: boolean, blocks: number | null = null) =>
      ({ revision, fingerprint: `native-${revision}`, creature: id(creature), action_index: actionIndex, attack, blocks: blocks === null ? null : id(blocks) });
    const preview = request.revision === 12
      ? {
          kind: 'blockers', declarable: [{ creature: id(83), label: 'Otter-Penguin', can_block: [id(84)], must_block: false }],
          steps: [request.plan.blocks.length ? step(12, 83, 201, false, 84) : step(12, 83, 202, false)],
          refused: null, declared_attackers: [], declared_blocks: request.plan.blocks,
        }
      : {
          kind: 'attackers',
          declarable: [{ creature: id(83), label: 'Otter-Penguin', can_block: [], must_block: false },
            { creature: id(85), label: 'Tiger-Seal', can_block: [], must_block: false },
            ...Array.from({ length: moreTigers }, (_, i) => ({ creature: id(86 + i), label: 'Tiger-Seal', can_block: [], must_block: false }))].filter(creature => request.revision === 10 || creature.creature.entity === 85),
          steps: [
            ...(request.revision === 10 ? [step(10, 83, attacks.has(83) ? 101 : 102, attacks.has(83))] : []),
            step(11, 85, attacks.has(85) ? 103 : 104, attacks.has(85)),
          ],
          refused: null, declared_attackers: request.plan.attackers, declared_blocks: [],
        };
    await route.fulfill({ json: { ...request, viewer: 0, preview } });
  });
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation', table: tableSnapshot(), data: current.projection, actions: [],
          session_id: 'combat-session', resume_token: 'combat-resume-token',
          recovery: { ...structuredClone(boltFixture.recovery), reason: 'initial_connect', frame: current, presentation_tail: [],
            presentation_cursor: 0, accepted_commands: [], replay_cursor: current.revision },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      authority.commands.push(command);
      const revision = command.expected_revision;
      const stage: Stage = after[revision] ?? (revision === 10 ? 'tiger' : revision === 11 ? 'block' : 'priority');
      const next = frameAt(stage);
      const blocked = revision === 12 && command.offer_id === 201;
      if (blocked) next.projection.agent.battlefield[0].damage = 1;
      if (blocked) next.projection.opponent.battlefield[0].damage = 2;
      current = next;
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted',
        update: acceptedUpdate(command, next, blocked ? combatEvents(12) : []) }));
    });
  });
  return authority;
}

async function startCombat(page: Page, after: Partial<Record<number, Stage>> = {}, moreTigers = 0): Promise<Authority> {
  const authority = await installCombatAuthority(page, after, moreTigers);
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
  await expect(page.getByTestId('combat-draft')).toBeVisible();
  return authority;
}

const card = (page: Page, id: number) => page.locator(`.board [data-testid=board-card][data-object-id="${id}"]`);

async function press(page: Page, mode: 'pointer' | 'keyboard', id: number): Promise<void> {
  if (mode === 'pointer') await card(page, id).click();
  else { await card(page, id).focus(); await page.keyboard.press('Enter'); }
}

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`combat is drafted on the board by ${mode} and sent one creature at a time on confirm`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const authority = await startCombat(page);

    // Every creature that can attack is outlined, and the primary reads the whole plan.
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'can');
    await expect(card(page, 85)).toHaveAttribute('data-draft', 'can');
    await expect(card(page, 83)).toHaveAccessibleName(/can attack; press to add it to the attack/);
    await expect(page.getByTestId('primary-action')).toHaveText(/No attack/);

    // Drafting sends nothing: Otter-Penguin leans in, Tiger-Seal stays home.
    await press(page, mode, 83);
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'in');
    await expect(card(page, 83)).toHaveClass(/attacking/);
    await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with Otter-Penguin · not with Tiger-Seal');
    await press(page, mode, 85);
    await press(page, mode, 85);
    // The button fits one line; the prompt above names the plan.
    await expect(page.getByTestId('primary-action')).toHaveText(/Attack with 1/);
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.commands).toEqual([]);

    // Confirm: each Command answers the frame the preview predicted for it.
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.commands.length).toBe(2);
    expect(authority.commands.map(command => [command.expected_revision, command.prompt_id, command.offer_id])).toEqual([[10, 110, 101], [11, 111, 104]]);

    // Blocking: Otter-Penguin can block only the Cub, so pressing it drafts that block and draws its line.
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
    await expect(page.getByTestId('combat-draft')).toHaveAttribute('data-kind', 'blockers');
    await expect(card(page, 83)).toHaveAccessibleName(/can block; press, then the attacker it blocks/);
    await press(page, mode, 83);
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'in');
    await expect(page.locator('[data-testid=board-lines] [data-line-kind=block]')).toHaveCount(1);
    await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Badgermole Cub');
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.commands).toHaveLength(2);
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.commands.length).toBe(3);
    expect(authority.commands[2]).toMatchObject({ expected_revision: 12, prompt_id: 112, offer_id: 201, answers: [] });
    expect(errors).toEqual([]);
  });
}

test('a frame the preview did not predict stops the sequence and redraws the board', async ({ page }) => {
  // After Otter-Penguin's attack the game asks for priority, not about Tiger-Seal.
  const authority = await startCombat(page, { 10: 'priority' });
  await card(page, 83).click();
  await card(page, 85).click();
  await expect(page.getByTestId('primary-action')).toHaveText(/Attack with 2/);
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
  await page.waitForTimeout(300);
  expect(authority.commands.map(command => command.offer_id)).toEqual([101]);
  await expect(page.getByTestId('combat-stopped')).toHaveText(/Stopped after 1 of 2: the game asked something the preview did not predict/);
  // The board is the frame's own again: no draft, Tiger-Seal not drawn attacking.
  await expect(page.getByTestId('combat-draft')).toHaveCount(0);
  await expect(card(page, 85)).not.toHaveClass(/attacking/);
  await expect(card(page, 85)).not.toHaveAttribute('data-draft');
});

test('a pile of identical creatures takes a count', async ({ page }) => {
  const authority = await startCombat(page, {}, 2);
  const pile = page.getByTestId('pile-count');
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 0 of 3 attacking');
  await pile.getByRole('button', { name: 'One more Tiger-Seal' }).click();
  await pile.getByRole('button', { name: 'One more Tiger-Seal' }).click();
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 2 of 3 attacking');
  await expect(page.locator('.board [data-testid=board-card][data-draft=in]')).toHaveCount(2);
  await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with 2 · not with Otter-Penguin and Tiger-Seal');
  expect(authority.commands).toEqual([]);
});

test('a reload mid-draft keeps the draft and commits nothing', async ({ page }) => {
  const authority = await startCombat(page);
  await card(page, 85).click();
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await page.reload();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await expect(card(page, 83)).toHaveAttribute('data-draft', 'can');
  await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with Tiger-Seal · not with Otter-Penguin');
  expect(authority.commands).toEqual([]);
  // The restored draft is sent as drafted.
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.keyboard.press('Space');
  await expect.poll(() => authority.commands.map(command => command.offer_id)).toEqual([102, 103]);
});

test('after combat, the block stays drawn and damage shows on the cards it landed on', async ({ page }, testInfo) => {
  const authority = await startCombat(page);
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
  await card(page, 83).click();
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
  // No frame holds the block any longer; the block events since the move draw it.
  await expect(page.locator('[data-testid=board-lines] [data-line-kind=block]')).toHaveCount(1);
  await expect(card(page, 84).getByTestId('damage-dealt')).toHaveText('−2');
  await expect(card(page, 83).getByTestId('damage-dealt')).toHaveText('−1');
  await expect(card(page, 84)).toHaveAccessibleName(/dealt 2 since your last move/);
  await expect(page.getByTestId('recent-fact')).toHaveText([
    'Otter-Penguin blocked Badgermole Cub',
    'Badgermole Cub dealt 1 to Otter-Penguin',
    'Otter-Penguin dealt 2 to Badgermole Cub',
  ]);
  await page.screenshot({ path: testInfo.outputPath('after-combat.png') });
  // Passing is the person's next move: the marks of the last one go.
  const sent = authority.commands.length;
  await page.keyboard.press('Space');
  await expect.poll(() => authority.commands.length).toBe(sent + 1);
});

test('Replay keeps playing through a live producer save and resumes at its saved choice', async ({ page }) => {
  const observation = combatObservation();
  const latest = {key: 'latest', kind: 'state', turn: 4, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false};
  const choice = {...latest, key: 'saved-choice', kind: 'choice'};
  const consequences = {...latest, key: 'saved-consequences', kind: 'consequences'};
  let produced = 0;
  let committed = false;
  let release: (() => void) | null = null;
  await page.route('**/api/traces', route => route.fulfill({json: []}));
  await page.route('**/api/watch/pending/step', async route => {
    produced++;
    await new Promise<void>(resolve => release = resolve);
    committed = true;
    await route.fulfill({json: {}});
  });
  await page.route('**/api/traces/pending/recording', route => route.fulfill({json: {
    id: 'pending', positions: committed ? [choice, consequences, latest] : [latest],
    total: committed ? 3 : 1, next_offset: null, revision: committed ? 12 : 11,
    status: 'waiting', can_produce: true, winner: null, players: [], source_kind: null, watch: true, can_feedback: false,
  }}));
  await page.route('**/api/traces/pending/positions/**', route => {
    const key = decodeURIComponent(new URL(route.request().url()).pathname.split('/positions/')[1]);
    const position = [choice, consequences, latest].find(position => position.key === key)!;
    return route.fulfill({json: {...position, observation, decision: null, outputs: null,
      description: position.kind === 'choice' ? 'Saved producer choice' : null, note_key: key, viewer: 0, revealed: false, history: [], history_labels: [], history_before: null,
      cursor: 0, transition_start: 0, consequences_recorded: true,
    }});
  });
  await page.goto('/replay?trace=pending');
  await expect(page.getByTestId('recording-status')).toHaveText('Waiting at the live edge');
  await page.clock.install({time: 0});
  await page.clock.pauseAt(1000);
  await page.getByRole('combobox', {name: 'Speed', exact: true}).selectOption('16');
  await page.getByRole('button', {name: 'Play', exact: true}).click();
  await page.clock.runFor(100);
  await expect.poll(() => release !== null).toBe(true);
  await expect(page.getByRole('button', {name: 'Pause', exact: true})).toBeEnabled();
  release!();
  const counter = page.getByTestId('position-counter');
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByTestId('recording-status')).toHaveText('Live Watch · reading saved recording');
  await expect.poll(() => new URL(page.url()).searchParams.get('position')).toBe('saved-choice');
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await page.clock.runFor(500);
  expect(produced).toBe(1);
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByTestId('game-log').locator('[aria-current="true"]')).toHaveCount(1);
  await page.reload();
  await expect(counter).toHaveText('Position 1 / 3');
  expect(produced).toBe(1);
});
