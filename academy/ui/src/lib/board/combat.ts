/**
 * A whole combat declaration, drafted on the board before any of it is sent.
 *
 * The engine asks about one creature at a time. The authority sends a live
 * `CombatGroup` descriptor (`declare_attackers` or `declare_blockers`) naming
 * every declarable creature and already-committed declarations. As a person
 * drafts on the board, Academy asks `/group-preview` (read-only) to validate
 * the whole plan against the real rules, and Confirm submits one atomic
 * `group_command`. Nothing here is a rules solver.
 *
 * A draft names creatures by their display ids (`PermanentState.id`, the
 * group descriptor's `ObjectRenderId.entity`).
 */
import type {
  AttackerItem,
  CombatGroup,
  DeclareAttackersPlan,
  DeclareBlockersPlan,
  GroupAddress,
} from '$lib/group';
import type { ExperienceFrame, ObjectRenderId } from '$lib/types';
import { PRIMARY_CHARS } from './prompt';

// ———————————————————————————————————————— Drafts

/** Attackers in the attack, or blocker → attacker; display ids. */
export interface Draft {
  attackers: number[];
  blocks: [number, number][];
}

export const EMPTY_DRAFT: Draft = { attackers: [], blocks: [] };

export function addressOf(frame: ExperienceFrame): GroupAddress | null {
  return frame.prompt
    ? {
        match_id: frame.match_id,
        revision: frame.revision,
        prompt_id: frame.prompt.id,
        frame_hash: frame.frame_hash,
      }
    : null;
}

/** Add the creatures caught by a sweep, keeping earlier selections and their order. */
export function addAttackers(draft: Draft, hits: ReadonlySet<number>): Draft {
  const attackers = new Set([...draft.attackers, ...hits]);
  return attackers.size === draft.attackers.length ? draft : { ...draft, attackers: [...attackers] };
}

export function toggleAttacker(draft: Draft, id: number): Draft {
  const attackers = draft.attackers.includes(id)
    ? draft.attackers.filter(other => other !== id)
    : [...draft.attackers, id];
  return { ...draft, attackers };
}

/** Place or remove one blocker, leaving every other assignment intact. */
export function setBlock(draft: Draft, blocker: number, attacker: number | null): Draft {
  if ((draft.blocks.find(([id]) => id === blocker)?.[1] ?? null) === attacker) return draft;
  const blocks = draft.blocks.filter(([id]) => id !== blocker);
  if (attacker !== null) blocks.push([blocker, attacker]);
  return { ...draft, blocks };
}

export function toggleBlock(draft: Draft, blocker: number, attacker: number): Draft {
  const assigned = draft.blocks.some(([b, a]) => b === blocker && a === attacker);
  return setBlock(draft, blocker, assigned ? null : attacker);
}

/** Draft the first `count` of a pile's identical creatures into the attack. */
export function setPileCount(draft: Draft, pile: number[], count: number): Draft {
  const rest = draft.attackers.filter(id => !pile.includes(id));
  return { ...draft, attackers: [...rest, ...pile.slice(0, Math.max(0, Math.min(pile.length, count)))] };
}

/** Filter a restored draft so it names only creatures currently declarable in `group`. */
export function sanitizeCombatDraft(saved: unknown, group: CombatGroup): Draft {
  if (!saved || typeof saved !== 'object') return EMPTY_DRAFT;
  const candidate = saved as Partial<Draft>;
  if (group.kind === 'declare_attackers') {
    const allowed = new Set(group.declarable.map(creature => creature.item.entity));
    const seen = new Set<number>();
    const attackers: number[] = [];
    for (const entity of Array.isArray(candidate.attackers) ? candidate.attackers : []) {
      if (typeof entity === 'number' && allowed.has(entity) && !seen.has(entity)) {
        seen.add(entity);
        attackers.push(entity);
      }
    }
    return { attackers, blocks: [] };
  }
  const canBlock = new Map(
    group.declarable.map(creature => [
      creature.item.entity,
      new Set(creature.can_block.map(attacker => attacker.entity)),
    ]),
  );
  const seen = new Set<number>();
  const blocks: [number, number][] = [];
  for (const entry of Array.isArray(candidate.blocks) ? candidate.blocks : []) {
    if (!Array.isArray(entry) || entry.length !== 2) continue;
    const [blocker, attacker] = entry;
    if (
      typeof blocker === 'number'
      && typeof attacker === 'number'
      && !seen.has(blocker)
      && canBlock.get(blocker)?.has(attacker)
    ) {
      seen.add(blocker);
      blocks.push([blocker, attacker]);
    }
  }
  return { attackers: [], blocks };
}

/** The typed group plan to send, using the descriptor's own exact identities. */
export function combatPlanOf(
  draft: Draft,
  group: CombatGroup,
): DeclareAttackersPlan | DeclareBlockersPlan {
  const ids = new Map<number, ObjectRenderId>();
  if (group.kind === 'declare_attackers') {
    for (const creature of group.declarable) ids.set(creature.item.entity, creature.item);
    const id = (entity: number): ObjectRenderId => ids.get(entity) ?? { entity, incarnation: 0 };
    return { kind: 'declare_attackers', attackers: draft.attackers.map(id) };
  }
  for (const creature of group.declarable) {
    ids.set(creature.item.entity, creature.item);
    for (const attacker of creature.can_block) ids.set(attacker.entity, attacker);
  }
  const id = (entity: number): ObjectRenderId => ids.get(entity) ?? { entity, incarnation: 0 };
  return {
    kind: 'declare_blockers',
    blocks: draft.blocks.map(([blocker, attacker]) => ({ blocker: id(blocker), attacker: id(attacker) })),
  };
}

/** What the engine already committed before this step (must-attack, skip_trivial, or raw single-step). */
export function committedFromGroup(group: CombatGroup): Draft {
  if (group.kind === 'declare_attackers') {
    return { attackers: group.committed.map(item => item.entity), blocks: [] };
  }
  return {
    attackers: [],
    blocks: group.committed.map(block => [block.blocker.entity, block.attacker.entity]),
  };
}

/** The board as drafted: drafted attackers lean, drafted blockers draw their line. Presentation only. */
export function drawDraft(frame: ExperienceFrame, draft: Draft): ExperienceFrame {
  if (!draft.attackers.length && !draft.blocks.length) return frame;
  const attacking = new Set(draft.attackers);
  const blocking = new Map(draft.blocks);
  const draw = (permanent: ExperienceFrame['projection']['agent']['battlefield'][number]) =>
    attacking.has(permanent.id)
      ? { ...permanent, attacking: true }
      : blocking.has(permanent.id)
        ? { ...permanent, blocking: blocking.get(permanent.id) }
        : permanent;
  const { agent, opponent } = frame.projection;
  return {
    ...frame,
    projection: {
      ...frame.projection,
      agent: { ...agent, battlefield: agent.battlefield.map(draw) },
      opponent: { ...opponent, battlefield: opponent.battlefield.map(draw) },
    },
  };
}

// ———————————————————————————————————————— Words

/** "Tiger-Seal", "Ally ×2": names in first-seen order, repeats counted. */
function counted(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name));
}

function list(parts: string[]): string {
  return parts.length <= 1 ? parts.join('') : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}`;
}

export const BLOCK_INSTRUCTION = 'Drag a blocker onto an attacker, or click each';
export const ATTACK_INSTRUCTION = 'Drag across creatures, or click each';

export interface DraftWords {
  /** The primary button: "Attack with 3", "No attack". */
  text: string;
  /** The whole plan: "Attack with 3 · not with Tiger-Seal". */
  plan: string;
}

/** Words for the declaration as it will stand: already committed plus the draft. */
export function draftWords(
  group: CombatGroup,
  drafted: Draft,
  names: Map<number, string>,
): DraftWords {
  const name = (id: number): string => names.get(id) ?? 'a creature';
  const committed = committedFromGroup(group);
  const draft: Draft = {
    attackers: [...committed.attackers.filter(id => !drafted.attackers.includes(id)), ...drafted.attackers],
    blocks: [
      ...committed.blocks.filter(([blocker]) => !drafted.blocks.some(([b]) => b === blocker)),
      ...drafted.blocks,
    ],
  };
  if (group.kind === 'declare_attackers') {
    const count = draft.attackers.length;
    if (count === 0) return { text: 'No attack', plan: `No attack · ${ATTACK_INSTRUCTION}` };
    const out = group.declarable
      .filter(creature => !draft.attackers.includes(creature.item.entity))
      .map(creature => creature.label);
    const one = `Attack with ${name(draft.attackers[0])}`;
    const text = count === 1 && one.length <= PRIMARY_CHARS ? one : `Attack with ${count}`;
    if (!out.length) {
      return {
        text,
        plan: count > 1 ? `Attack with all ${count}` : `Attack with ${name(draft.attackers[0])}`,
      };
    }
    const left = counted(out);
    return {
      text,
      plan: `${count === 1 ? `Attack with ${name(draft.attackers[0])}` : text} · not with ${left.length > 2 ? `${out.length} others` : list(left)}`,
    };
  }
  const count = draft.blocks.length;
  if (count === 0) return { text: 'No blocks', plan: `No blocks · ${BLOCK_INSTRUCTION}` };
  const byAttacker = new Map<number, string[]>();
  for (const [blocker, attacker] of draft.blocks) {
    byAttacker.set(attacker, [...(byAttacker.get(attacker) ?? []), name(blocker)]);
  }
  const lines = [...byAttacker].map(
    ([attacker, blockers]) => `${list(counted(blockers))} on ${name(attacker)}`,
  );
  return { text: `Block with ${count}`, plan: lines.join(' · ') };
}

/** Identical declarable creatures drawn as one pile: "4 of 9 Allies". */
export interface PileCount {
  name: string;
  ids: number[];
  drafted: number;
}

export function pileCounts(
  declarable: readonly AttackerItem[],
  draft: Draft,
  pileOf: (id: number) => string | null,
): PileCount[] {
  const groups = new Map<string, PileCount>();
  for (const creature of declarable) {
    const key = pileOf(creature.item.entity);
    if (key === null) continue;
    const entry = groups.get(key) ?? { name: creature.label, ids: [], drafted: 0 };
    entry.ids.push(creature.item.entity);
    if (draft.attackers.includes(creature.item.entity)) entry.drafted += 1;
    groups.set(key, entry);
  }
  return [...groups.values()].filter(entry => entry.ids.length > 1);
}
