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

function combatObservation(): Observation {
  const observation = structuredClone(boltFixture.recovery.frame.projection);
  observation.turn = {
    turn_number: 4,
    phase: 'COMBAT',
    step: 'DECLARE_ATTACKERS_STEP',
    active_player_id: 0,
    agent_player_id: 0,
  };
  observation.agent.battlefield = [{
    id: 83,
    name: 'Otter-Penguin',
    controller_id: 0,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 2,
    toughness: 2,
    base_power: 2,
    base_toughness: 2,
    plus1_counters: 0,
  }];
  observation.opponent.battlefield = [{
    id: 84,
    name: 'Badgermole Cub',
    controller_id: 1,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 1,
    toughness: 1,
    base_power: 1,
    base_toughness: 1,
    plus1_counters: 0,
  }];
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

function frameAt(stage: 'attack' | 'block' | 'priority'): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const revision = stage === 'attack' ? 10 : stage === 'block' ? 11 : 12;
  const observation = combatObservation();
  const actionSpace = stage === 'attack' ? 'DECLARE_ATTACKER' : stage === 'block' ? 'DECLARE_BLOCKER' : 'PRIORITY';
  if (stage === 'block') {
    observation.turn = { ...observation.turn, active_player_id: 1, step: 'DECLARE_BLOCKERS_STEP' };
    observation.opponent.battlefield[0].attacking = true;
  }
  if (stage === 'priority') observation.turn = { ...observation.turn, active_player_id: 1, phase: 'COMBAT', step: 'COMBAT_END' };
  const offers = stage === 'attack'
    ? [
        offer(101, 'declare_attackers', 'Attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'source', object: OTTER }]),
        offer(102, 'declare_attackers', 'Do not attack with Otter-Penguin', 'DECLARE_ATTACKER', [83], [{ role: 'choice', object: OTTER }]),
      ]
    : stage === 'block'
      ? [
          offer(201, 'declare_blockers', 'Block Badgermole Cub with Otter-Penguin', 'DECLARE_BLOCKER', [83, 84],
            [{ role: 'choice', object: OTTER }, { role: 'target', object: CUB }]),
          offer(202, 'declare_blockers', 'Otter-Penguin: do not block', 'DECLARE_BLOCKER', [83], [{ role: 'choice', object: OTTER }]),
        ]
      : [offer(301, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', [])];
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

/** Attack, then block, then the combat's end; each Command answers the frame it was made at. */
async function installCombatAuthority(page: Page, commands: Command[]): Promise<void> {
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command };
      if (message.type === 'new_game') {
        const frame = frameAt('attack');
        socket.send(JSON.stringify({
          type: 'observation', table: tableSnapshot(), data: frame.projection, actions: [],
          session_id: 'combat-session', resume_token: 'combat-resume-token',
          recovery: { ...structuredClone(boltFixture.recovery), reason: 'initial_connect', frame, presentation_tail: [],
            presentation_cursor: 0, accepted_commands: [], replay_cursor: frame.revision },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) return;
      const command = message.command;
      commands.push(command);
      const blocked = command.expected_revision === 11;
      const next = command.expected_revision === 10 ? frameAt('block') : frameAt('priority');
      if (blocked) next.projection.agent.battlefield[0].damage = 1;
      if (blocked) next.projection.opponent.battlefield[0].damage = 2;
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted',
        update: acceptedUpdate(command, next, blocked ? combatEvents(11) : []) }));
    });
  });
}

async function startCombat(page: Page, commands: Command[]): Promise<void> {
  await installCombatAuthority(page, commands);
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
}

const card = (page: Page, id: number) => page.locator(`.board [data-testid=table-card][data-object-id="${id}"]`);

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`combat is answered on the board by ${mode}: attack with the creature, block on the attacker`, async ({ page }) => {
    const commands: Command[] = [];
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await startCombat(page, commands);

    // The creature asked about is lifted and is the attack; declining is a separate button.
    await expect(card(page, 83)).toHaveClass(/asked/);
    await expect(card(page, 83)).toHaveClass(/legal/);
    await expect(card(page, 83)).toHaveAccessibleName(/Attack with Otter-Penguin \(1\)$/);
    await expect(page.getByTestId('prompt-action')).toHaveText(['2Do not attack with Otter-Penguin']);
    await expect(page.getByTestId('action-option')).toHaveCount(2);
    if (mode === 'pointer') await card(page, 83).click();
    else await page.keyboard.press('1');
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toMatchObject({ match_id: 'combat-prompt-proof', expected_revision: 10, prompt_id: 110, offer_id: 101, answers: [] });

    // Blocking: the attacker is the control; the blocker asked about is lifted; not blocking is a button.
    await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
    await expect(card(page, 83)).toHaveClass(/asked/);
    await expect(card(page, 84)).toHaveClass(/legal/);
    await expect(card(page, 84)).toHaveAccessibleName(/Block Badgermole Cub with Otter-Penguin \(1\)$/);
    await expect(page.getByTestId('prompt-action')).toHaveText(['2Otter-Penguin: do not block']);
    if (mode === 'pointer') await card(page, 84).click();
    else await page.keyboard.press('1');
    await expect.poll(() => commands.length).toBe(2);
    expect(commands[1]).toMatchObject({ expected_revision: 11, prompt_id: 111, offer_id: 201, answers: [] });
    expect(errors).toEqual([]);
  });
}

test('after combat, the block stays drawn and damage shows on the cards it landed on', async ({ page }, testInfo) => {
  const commands: Command[] = [];
  await startCombat(page, commands);
  await card(page, 83).click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
  await card(page, 84).click();
  await expect(page.getByTestId('table')).toHaveAttribute('data-action-space', 'PRIORITY');
  // No frame holds the block any longer; the block events since the move draw it.
  await expect(page.locator('[data-testid=table-lines] [data-line-kind=block]')).toHaveCount(1);
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
  await page.keyboard.press('Space');
  await expect.poll(() => commands.length).toBe(3);
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
