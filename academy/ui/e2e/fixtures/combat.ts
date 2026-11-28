import { expect, type Page } from '@playwright/test';
import { gameReply } from './game';
import boltFixtureJson from '../../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  DeclareAttackersDecisionGroup,
  DeclareBlockersDecisionGroup,
  DecisionGroupCommand,
  DecisionGroupPlan,
  DecisionGroupRefusal,
  LiveDecisionGroup,
} from '../../src/lib/decision-group';
import type {
  Command,
  ExperienceFrame,
  FrameUpdate,
  InteractionOffer,
  Observation,
  PresentationEvent,
  RecoveryEnvelope,
} from '../../src/lib/types';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';
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

const id = (entity: number) => ({ entity, incarnation: 0 });

function liveDecisionGroupFor(frame: ExperienceFrame, moreTigers = 0): LiveDecisionGroup | null {
  if (!frame.prompt) return null;
  const address = {
    match_id: frame.match_id,
    revision: frame.revision,
    prompt_id: frame.prompt.id,
    frame_hash: frame.frame_hash,
  };
  if (frame.action_space === 'DECLARE_ATTACKER') {
    const declarable: DeclareAttackersDecisionGroup['declarable'] = [
      { item: id(83), label: 'Otter-Penguin', must_attack: false },
      { item: id(85), label: 'Tiger-Seal', must_attack: false },
      ...Array.from({ length: moreTigers }, (_, i) => ({
        item: id(86 + i),
        label: 'Tiger-Seal',
        must_attack: false,
      })),
    ].filter(creature => frame.revision === 10 || creature.item.entity !== 83);
    const committed = frame.revision === 11 ? [id(83)] : [];
    return { ...address, kind: 'declare_attackers', declarable, committed };
  }
  if (frame.action_space === 'DECLARE_BLOCKER') {
    const group: DeclareBlockersDecisionGroup = {
      ...address,
      kind: 'declare_blockers',
      declarable: [{ item: id(83), label: 'Otter-Penguin', can_block: [id(84)], must_block: false }],
      committed: [],
    };
    return group;
  }
  return null;
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

function acceptedUpdate(
  command: { command_id: string; expected_revision: number },
  frame: ExperienceFrame,
  presentation: PresentationEvent[] = [],
): FrameUpdate {
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

export interface CombatAuthority {
  commands: Command[];
  groupCommands: DecisionGroupCommand[];
  previews: { revision: number; plan: DecisionGroupPlan }[];
}

interface CombatScenario {
  moreTigers?: number;
  refuseGroupOnce?: DecisionGroupRefusal;
  finishWork?: Promise<void>;
}

/**
 * A combat authority: sends live `group` descriptors, previews drafts on
 * `/decision-group-preview` (without steps), and executes whole declarations on one
 * atomic `decision_group_command`. Raw single-step `command`s still work too.
 */
export async function startCombat(
  page: Page,
  { moreTigers = 0, refuseGroupOnce, finishWork }: CombatScenario = {},
): Promise<CombatAuthority> {
  const authority: CombatAuthority = { commands: [], groupCommands: [], previews: [] };
  let current = frameAt('otter', moreTigers);
  let pendingRefusal = refuseGroupOnce ?? null;

  await page.route('**/api/games/*/decision-group-preview', async route => {
    const request = route.request().postDataJSON() as {
      match_id: string; revision: number; prompt_id: number; frame_hash: string; plan: DecisionGroupPlan;
    };
    authority.previews.push({ revision: request.revision, plan: request.plan });
    const outcome = request.plan.kind === 'declare_blockers'
      ? { kind: 'declare_blockers', blocks: request.plan.blocks }
      : request.plan.kind === 'declare_attackers'
        ? { kind: 'declare_attackers', attackers: request.plan.attackers }
        : null;
    await route.fulfill({
      json: {
        match_id: request.match_id,
        revision: request.revision,
        prompt_id: request.prompt_id,
        frame_hash: request.frame_hash,
        viewer: 0,
        complete: true,
        refused: null,
        outcome,
      },
    });
  });
  const game = gameReply();
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage(async raw => {
      const message = JSON.parse(String(raw)) as {
        type?: string;
        command?: Command | DecisionGroupCommand;
      };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation', ...game, data: current.projection, actions: [],
          decision_group: liveDecisionGroupFor(current, moreTigers),
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

      if (message.type === 'decision_group_command' && message.command) {
        const command = message.command as DecisionGroupCommand;
        authority.groupCommands.push(command);
        const revision = command.expected_revision;
        if (pendingRefusal) {
          const refused = pendingRefusal;
          pendingRefusal = null;
          socket.send(JSON.stringify({
            type: 'command_outcome',
            status: 'rejected',
            decision_group: liveDecisionGroupFor(current, moreTigers),
            rejection: {
              command_id: command.command_id,
              code: 'decision_group_refused',
              message: refused.reason,
              current_revision: revision,
              current_prompt: current.prompt!.id,
              refused,
            },
          }));
          return;
        }
        const stage: Stage = command.plan.kind === 'declare_attackers' ? 'block' : 'priority';
        const next = frameAt(stage, moreTigers);
        const blocked = command.plan.kind === 'declare_blockers' && command.plan.blocks.length > 0;
        if (blocked) {
          next.projection.agent.battlefield[0].damage = 1;
          next.projection.opponent.battlefield[0].damage = 2;
        }
        await (finishWork ?? Promise.resolve());
        current = next;
        const update = acceptedUpdate(command, next, blocked ? combatEvents(12) : []);
        const work = (state: 'running' | 'complete') => ({
          operation_id: `advance-${revision}`, sequence: revision, kind: 'advance', started_at: 0, state,
          scope: 'match', setup_id: null, loading_seat: null, match_id: next.match_id, revision: next.revision, chooser: null,
        });
        socket.send(JSON.stringify({
          type: 'command_outcome', status: 'accepted', update, work: work('complete'),
          decision_group: liveDecisionGroupFor(next, moreTigers),
        }));
        return;
      }

      if (message.type === 'command' && message.command) {
        const command = message.command as Command;
        authority.commands.push(command);
        const revision = command.expected_revision;
        const stage: Stage = revision === 10 ? 'tiger' : revision === 11 ? 'block' : 'priority';
        const next = frameAt(stage, moreTigers);
        current = next;
        const update = acceptedUpdate(command, next);
        socket.send(JSON.stringify({
          type: 'command_outcome', status: 'accepted', update,
          decision_group: liveDecisionGroupFor(next, moreTigers),
        }));
      }
    });
  });
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
  await expect(page.getByTestId('combat-draft')).toBeVisible();
  return authority;
}

/**
 * A live group folds each per-step raw offer under a closed disclosure.
 * Scripted players that pick raw offers open it whenever the disclosure is
 * attached (combat declarations and library arrangements alike); prompts
 * without a live group already list raw offers directly.
 */
export async function showEachCreaturesChoice(page: Page): Promise<void> {
  const raw = page.getByTestId('per-creature-choices');
  if (await raw.count() === 0) return;
  // A first click during an opponent read-out ends the read-out; click again until the disclosure opens.
  await expect(async () => {
    if (await raw.count() === 0) return;
    if (!(await raw.evaluate(details => (details as HTMLDetailsElement).open))) await raw.locator('summary').click();
    if (await raw.count() > 0) await expect(raw).toHaveJSProperty('open', true);
  }).toPass({ timeout: 10_000 });
}
