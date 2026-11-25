/**
 * Structured combat breakdown table for large combats (three or more creatures
 * in the fight: attackers + blockers). Small combats keep the single-line walk
 * caption; big combats compose one row per attacker showing who blocked it,
 * what it dealt and took, and what died.
 */
import type { Observation, PermanentState, PresentationEvent, PresentationLabels } from '$lib/types';
import { eventObjectName, playerSide, tellRule, type TellPlayers } from './tell';
import { turnAt } from './history';

/** Minimum number of participating creatures (attackers + blockers) required to show the table. */
export const COMBAT_TABLE_MIN_CREATURES = 3;

export interface CombatParticipant {
  entity: number;
  name: string;
  pt: string | null;
}

export interface CombatRow {
  attacker: CombatParticipant;
  blockers: CombatParticipant[];
  dealt: string;
  took: string;
  result: string;
  /** Entity ids involved in this exchange (hovering a row raises them on the board). */
  subjects: number[];
  /** Representative event seq for this attacker's exchange. */
  seq: number;
}

export interface CombatTable {
  turn: number | null;
  attackerSide: 'you' | 'opponent';
  heading: string;
  creatureCount: number;
  rows: CombatRow[];
}

function permanentPt(
  entity: number,
  projections: readonly (Observation | null | undefined)[],
): string | null {
  for (const proj of projections) {
    if (!proj) continue;
    const perm: PermanentState | undefined =
      proj.agent.battlefield.find(p => p.id === entity)
      ?? proj.opponent.battlefield.find(p => p.id === entity);
    if (perm && perm.power !== null && perm.toughness !== null) {
      return `${perm.power}/${perm.toughness}`;
    }
  }
  return null;
}

function formatName(
  entity: number,
  knownNames: ReadonlyMap<number, string>,
): string {
  return knownNames.get(entity) ?? 'Creature';
}

/**
 * Build a combat table when `activeSeqs` rests on a combat damage or combat
 * death stop involving at least `COMBAT_TABLE_MIN_CREATURES` creatures.
 * Returns null for non-combat stops or small combats (< 3 creatures).
 */
export function buildCombatTable(
  events: readonly PresentationEvent[],
  activeSeqs: ReadonlySet<number>,
  labelsAt: (index: number) => PresentationLabels,
  players: TellPlayers,
  projection: Observation,
  priorProjection?: Observation | null,
): CombatTable | null {
  if (activeSeqs.size === 0) return null;

  // Locate the latest event in `activeSeqs` to anchor the active combat window.
  let maxActiveIdx = -1;
  let minActiveIdx = -1;
  for (let i = 0; i < events.length; i += 1) {
    if (activeSeqs.has(events[i].seq)) {
      if (minActiveIdx < 0) minActiveIdx = i;
      maxActiveIdx = i;
    }
  }
  if (maxActiveIdx < 0) return null;

  // Find the most recent `attack_group` in the same turn at or before `maxActiveIdx`.
  let attackIdx = -1;
  for (let i = maxActiveIdx; i >= 0; i -= 1) {
    const kind = events[i].kind.kind;
    if (kind === 'turn_started') break;
    if (kind === 'attack_group') {
      attackIdx = i;
      break;
    }
  }
  if (attackIdx < 0) return null;

  const attackEvent = events[attackIdx];
  if (attackEvent.kind.kind !== 'attack_group') return null;

  // Scan from `attack_group` until the next turn start or next `attack_group`.
  let endIdx = events.length - 1;
  for (let i = attackIdx + 1; i < events.length; i += 1) {
    const kind = events[i].kind.kind;
    if (kind === 'turn_started' || kind === 'attack_group') {
      endIdx = i - 1;
      break;
    }
  }

  const attackers = attackEvent.kind.attackers;
  const attackerSet = new Set(attackers.map(a => a.entity));
  const defenderSide = attackEvent.kind.defender.kind === 'player'
    ? playerSide(attackEvent.kind.defender.id, players.youSeat)
    : null;
  const attackerSide: 'you' | 'opponent' = defenderSide === 'you' ? 'opponent' : 'you';
  const defenderName = defenderSide === 'you'
    ? (players.you === 'You' ? 'you' : players.you)
    : players.opponent;

  const knownNames = new Map<number, string>();
  const recordName = (event: PresentationEvent, index: number, obj: { entity: number; incarnation: number }): void => {
    if (knownNames.has(obj.entity)) return;
    const name = eventObjectName(event, obj, labelsAt(index));
    if (name) knownNames.set(obj.entity, name);
  };
  for (const attacker of attackers) recordName(attackEvent, attackIdx, attacker);

  const blockersByAttacker = new Map<number, number[]>();
  const blockerSet = new Set<number>();
  for (let i = attackIdx + 1; i <= endIdx; i += 1) {
    const event = events[i];
    if (event.kind.kind !== 'blocked') continue;
    recordName(event, i, event.kind.attacker);
    const list = blockersByAttacker.get(event.kind.attacker.entity) ?? [];
    for (const blocker of event.kind.blockers) {
      recordName(event, i, blocker);
      if (!list.includes(blocker.entity)) list.push(blocker.entity);
      blockerSet.add(blocker.entity);
    }
    blockersByAttacker.set(event.kind.attacker.entity, list);
  }

  const combatants = new Set<number>([...attackerSet, ...blockerSet]);
  if (combatants.size < COMBAT_TABLE_MIN_CREATURES) return null;

  // Identify combat damage events (source is a combatant) and deaths caused by them.
  const combatDamageSeqs = new Set<number>();
  const combatEventsInActive = { hit: false };
  const dealtBySource = new Map<number, { targetName: string; targetEntity: number | null; amount: number; isPlayer: boolean }[]>();
  const tookByTarget = new Map<number, number>();
  const diedRule = new Map<number, string>();
  const firstSeqForAttacker = new Map<number, number>();

  for (let i = attackIdx + 1; i <= endIdx; i += 1) {
    const event = events[i];
    const kind = event.kind;
    if (kind.kind === 'damage' && kind.source?.kind === 'object' && combatants.has(kind.source.id.entity)) {
      const srcEntity = kind.source.id.entity;
      recordName(event, i, kind.source.id);
      combatDamageSeqs.add(event.seq);
      if (activeSeqs.has(event.seq)) combatEventsInActive.hit = true;

      if (kind.target.kind === 'object') {
        const tgtEntity = kind.target.id.entity;
        recordName(event, i, kind.target.id);
        const targetName = formatName(tgtEntity, knownNames);
        const list = dealtBySource.get(srcEntity) ?? [];
        list.push({ targetName, targetEntity: tgtEntity, amount: kind.amount, isPlayer: false });
        dealtBySource.set(srcEntity, list);
        tookByTarget.set(tgtEntity, (tookByTarget.get(tgtEntity) ?? 0) + kind.amount);
        if (attackerSet.has(srcEntity) && !firstSeqForAttacker.has(srcEntity)) {
          firstSeqForAttacker.set(srcEntity, event.seq);
        } else if (attackerSet.has(tgtEntity) && !firstSeqForAttacker.has(tgtEntity)) {
          firstSeqForAttacker.set(tgtEntity, event.seq);
        }
      } else if (kind.target.kind === 'player') {
        const list = dealtBySource.get(srcEntity) ?? [];
        list.push({ targetName: defenderName, targetEntity: null, amount: kind.amount, isPlayer: true });
        dealtBySource.set(srcEntity, list);
        if (attackerSet.has(srcEntity) && !firstSeqForAttacker.has(srcEntity)) {
          firstSeqForAttacker.set(srcEntity, event.seq);
        }
      }
    } else if (kind.kind === 'died') {
      const isCombatDeath = (event.cause !== undefined && combatDamageSeqs.has(event.cause))
        || (combatDamageSeqs.size > 0 && kind.objects.some(o => combatants.has(o.entity)));
      if (!isCombatDeath) continue;
      if (activeSeqs.has(event.seq)) combatEventsInActive.hit = true;
      const ruleText = event.rule ? tellRule(event.rule) : 'died';
      for (const obj of kind.objects) {
        recordName(event, i, obj);
        diedRule.set(obj.entity, ruleText);
      }
    }
  }

  // Only show the combat table on the combat damage / combat death stop(s).
  if (!combatEventsInActive.hit) return null;

  const projections = [priorProjection, projection];
  const rows: CombatRow[] = attackers.map(att => {
    const entity = att.entity;
    const name = formatName(entity, knownNames);
    const attacker: CombatParticipant = {
      entity,
      name,
      pt: permanentPt(entity, projections),
    };
    const blockerEntities = blockersByAttacker.get(entity) ?? [];
    const blockers: CombatParticipant[] = blockerEntities.map(bId => ({
      entity: bId,
      name: formatName(bId, knownNames),
      pt: permanentPt(bId, projections),
    }));

    const hits = dealtBySource.get(entity) ?? [];
    const dealt = hits.length === 0
      ? '0'
      : hits.map(h => (h.isPlayer ? `${h.amount} to ${h.targetName}` : `${h.amount} to ${h.targetName}`)).join(', ');

    const tookAmount = tookByTarget.get(entity) ?? 0;
    const took = String(tookAmount);

    const outcomes: string[] = [];
    const attDeath = diedRule.get(entity);
    if (attDeath) {
      outcomes.push(attDeath === 'died' ? `${name} died` : `died · ${attDeath}`);
    }
    for (const blocker of blockers) {
      const bDeath = diedRule.get(blocker.entity);
      if (bDeath) {
        outcomes.push(`${blocker.name} died`);
      }
    }
    const playerDmg = hits.filter(h => h.isPlayer).reduce((sum, h) => sum + h.amount, 0);
    if (playerDmg > 0) {
      outcomes.push(`${defenderName} −${playerDmg}`);
    }
    if (outcomes.length === 0) {
      outcomes.push(tookAmount > 0 ? 'survived' : '—');
    }

    return {
      attacker,
      blockers,
      dealt,
      took,
      result: outcomes.join(' · '),
      subjects: [entity, ...blockerEntities],
      seq: firstSeqForAttacker.get(entity) ?? attackEvent.seq,
    };
  });

  const turn = turnAt(events, attackEvent.seq);
  const attackerLabel = attackerSide === 'you'
    ? (players.you === 'You' ? 'Your' : `${players.you}'s`)
    : `${players.opponent}'s`;

  return {
    turn,
    attackerSide,
    heading: `${attackerLabel} attack${turn !== null ? ` · turn ${turn}` : ''} (${combatants.size} creatures)`,
    creatureCount: combatants.size,
    rows,
  };
}
