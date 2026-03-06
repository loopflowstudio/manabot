import type { GameLogEntry, Observation, PermanentState } from './types';

function playerLabel(side: 'agent' | 'opponent'): string {
  return side === 'agent' ? 'Hero' : 'Villain';
}

function battlefieldMap(permanents: PermanentState[]): Map<number, string> {
  return new Map(
    permanents.map((permanent) => [permanent.id, permanent.name ?? 'Unknown permanent']),
  );
}

function cardMap(cards: Array<{ id: number; name: string }>): Map<number, string> {
  return new Map(cards.map((card) => [card.id, card.name]));
}

function pushBattlefieldNotes(
  notes: string[],
  label: string,
  previous: PermanentState[],
  next: PermanentState[],
): void {
  const previousMap = battlefieldMap(previous);
  const nextMap = battlefieldMap(next);

  for (const [id, name] of nextMap) {
    if (!previousMap.has(id)) {
      notes.push(`${label}: ${name} entered the battlefield.`);
    }
  }

  for (const [id, name] of previousMap) {
    if (!nextMap.has(id)) {
      notes.push(`${label}: ${name} left the battlefield.`);
    }
  }
}

function pushZoneNotes(
  notes: string[],
  label: string,
  zone: string,
  previous: Array<{ id: number; name: string }>,
  next: Array<{ id: number; name: string }>,
): void {
  const previousMap = cardMap(previous);
  const nextMap = cardMap(next);

  for (const [id, name] of nextMap) {
    if (!previousMap.has(id)) {
      notes.push(`${label}: ${name} moved to ${zone}.`);
    }
  }
}

export function deriveObservationNotes(
  previous: Observation | null,
  next: Observation,
): string[] {
  if (previous === null) {
    return [];
  }

  const notes: string[] = [];

  for (const side of ['agent', 'opponent'] as const) {
    const previousPlayer = previous[side];
    const nextPlayer = next[side];
    const label = playerLabel(side);

    if (previousPlayer.life !== nextPlayer.life) {
      notes.push(`${label} life: ${previousPlayer.life} → ${nextPlayer.life}.`);
    }

    pushBattlefieldNotes(
      notes,
      label,
      previousPlayer.battlefield,
      nextPlayer.battlefield,
    );
    pushZoneNotes(
      notes,
      label,
      'graveyard',
      previousPlayer.graveyard,
      nextPlayer.graveyard,
    );
    pushZoneNotes(notes, label, 'stack', previousPlayer.stack, nextPlayer.stack);
  }

  return notes;
}

export function createLogEntry(
  id: string,
  actor: GameLogEntry['actor'],
  text: string,
  details: string[] = [],
): GameLogEntry {
  return { id, actor, text, details };
}
