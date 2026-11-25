import { expect, type Page } from '@playwright/test';
import { tableSnapshot } from './table';
import boltFixtureJson from '../../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  Command,
  ExperienceFrame,
  FrameUpdate,
  InteractionOffer,
  Observation,
  PresentationEvent,
  RecoveryEnvelope,
} from '../../src/lib/types';


const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

function permanent(id: number, name: string, controller: number, power: number, toughness: number): Observation['agent']['battlefield'][number] {
  return {
    id, name, controller_id: controller, tapped: false, damage: 0, summoning_sick: false,
    power, toughness, base_power: power, base_toughness: toughness, plus1_counters: 0,
  };
}

/** You: Otter-Penguin (83) and Tiger-Seal (85). The opponent: Badgermole Cub (84). */
export function combatObservation(): Observation {
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

interface CombatScenario {
  after?: Partial<Record<number, Stage>>;
  moreTigers?: number;
  rejectOnceAt?: number;
  finishWork?: Promise<void>;
}

/**
 * A combat authority: each Command answers the frame it was made at, and the
 * combat preview answers a drafted plan with the per-creature offers the
 * engine would ask for. `after` replaces the frame that follows a revision,
 * to stand for a decision the preview did not predict.
 */
export async function startCombat(page: Page, { after = {}, moreTigers = 0, rejectOnceAt, finishWork }: CombatScenario = {}): Promise<Authority> {
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
          kind: 'blockers', declarable: [{ creature: id(83), label: 'Otter-Penguin', can_block: [id(84)], must_block: false, must_attack: false }],
          steps: [request.plan.blocks.length ? step(12, 83, 201, false, 84) : step(12, 83, 202, false)],
          refused: null, declared_attackers: [], declared_blocks: request.plan.blocks,
        }
      : {
          kind: 'attackers',
          declarable: [{ creature: id(83), label: 'Otter-Penguin', can_block: [], must_block: false, must_attack: false },
            { creature: id(85), label: 'Tiger-Seal', can_block: [], must_block: false, must_attack: false },
            ...Array.from({ length: moreTigers }, (_, i) => ({ creature: id(86 + i), label: 'Tiger-Seal', can_block: [], must_block: false, must_attack: false }))].filter(creature => request.revision === 10 || creature.creature.entity === 85),
          steps: [
            ...(request.revision === 10 ? [step(10, 83, attacks.has(83) ? 101 : 102, attacks.has(83))] : []),
            step(11, 85, attacks.has(85) ? 103 : 104, attacks.has(85)),
          ],
          refused: null, declared_attackers: request.plan.attackers, declared_blocks: [],
        };
    await route.fulfill({ json: { ...request, viewer: 0, preview } });
  });
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage(async raw => {
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
      if (revision === rejectOnceAt) {
        rejectOnceAt = undefined;
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: command.command_id, code: 'stale_revision', message: 'The declaration was refused.',
            current_revision: revision, current_prompt: current.prompt!.id },
          recovery: { ...structuredClone(boltFixture.recovery), reason: 'command_rejected', frame: current, presentation_tail: [],
            presentation_cursor: 0, accepted_commands: [], replay_cursor: revision } }));
        return;
      }
      const stage: Stage = after[revision] ?? (revision === 10 ? 'tiger' : revision === 11 ? 'block' : 'priority');
      const next = frameAt(stage);
      const blocked = revision === 12 && command.offer_id === 201;
      if (blocked) next.projection.agent.battlefield[0].damage = 1;
      if (blocked) next.projection.opponent.battlefield[0].damage = 2;
      current = next;
      // As the server does: the accepted frame arrives while the table's work still runs, then again once it completes.
      const update = acceptedUpdate(command, next, blocked ? combatEvents(12) : []);
      const work = (state: 'running' | 'complete') => ({ operation_id: `advance-${revision}`, sequence: revision, kind: 'advance', started_at: 0, state,
        scope: 'match', setup_id: null, loading_seat: null, match_id: next.match_id, revision: next.revision, chooser: null });
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted', update, work: work('running') }));
      await (finishWork ?? new Promise(resolve => setTimeout(resolve, 50)));
      socket.send(JSON.stringify({ type: 'command_outcome', status: 'accepted', update, work: work('complete') }));
    });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
  await expect(page.getByTestId('combat-draft')).toBeVisible();
  return authority;
}

/**
 * A live combat draft folds each creature's own answer under a closed
 * disclosure. Scripted players that pick those answers open it once the draft
 * is drawn; a declaration without a draft (no preview) lists them already.
 */
export async function showEachCreaturesChoice(page: Page): Promise<void> {
  const kind = await page.getByTestId('action-panel').getAttribute('data-action-space-kind');
  if (kind !== 'DECLARE_ATTACKER' && kind !== 'DECLARE_BLOCKER') return;
  const raw = page.getByTestId('per-creature-choices');
  try { await raw.waitFor({ state: 'attached', timeout: 10_000 }); } catch { return; }
  if (!(await raw.evaluate(details => (details as HTMLDetailsElement).open))) await raw.locator('summary').click();
}
