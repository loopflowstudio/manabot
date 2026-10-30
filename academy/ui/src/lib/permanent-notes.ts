import { copyLabel } from './decision-labels';
import type { ContinuousEffect, PermanentState } from './types';

// One continuous-effect record, in the words the LLM text uses
// (`manabot/semantic/text_observation.py`, `effect_phrase`).
export function effectNote(effect: ContinuousEffect): string {
  const signed = (value: number | undefined): string =>
    `${(value ?? 0) >= 0 ? '+' : ''}${value ?? 0}`;
  const phrase = (() => {
    switch (effect.modification) {
      case 'ModifyPt':
        return `${signed(effect.power)}/${signed(effect.toughness)}`;
      case 'SetBasePt':
        return `base power/toughness ${effect.power ?? 'unchanged'}/${effect.toughness ?? 'unchanged'}`;
      case 'AddKeywords':
        return (effect.keywords ?? []).map(keyword => keyword.replaceAll('_', ' ')).join(', ');
      case 'AddSubtypes':
        return `is also ${(effect.subtypes ?? []).join(', ')}`;
      case 'BecomeArtifactCreature':
        return 'is an artifact creature';
      case 'LoseAllAbilities':
        return 'loses all abilities';
      case 'Firebending':
        return `firebending ${effect.amount}`;
      case 'CantBeBlocked':
        return "can't be blocked";
      case 'ExileIfWouldDie':
        return 'exiled instead if it would die';
      case 'KeepUnspentMana':
        return `keeps unspent ${(effect.color ?? '').toLowerCase()} mana`;
      case 'NoMaximumHandSize':
        return 'no maximum hand size';
      default:
        return effect.modification;
    }
  })();
  const until = effect.duration === 'EndOfTurn' ? 'until end of turn: ' : '';
  return `${until}${phrase}${effect.source ? ` (${effect.source})` : ''}`;
}

// Effect-derived state the observation carries for a permanent, as labels
// (Jack Heart, 2026-10-21: show the agent's observation, add nothing).
// `table` is every visible permanent, so attachments across seats resolve.
export function permanentNotes(permanent: PermanentState, table: PermanentState[]): string[] {
  const named = (id: number): string => {
    const other = table.find(candidate => candidate.id === id);
    if (!other) return 'an unseen permanent';
    const copy = copyLabel(table, id);
    return `${other.name ?? 'Unknown Permanent'}${copy ? ` (${copy})` : ''}`;
  };
  return [
    ...Object.entries(permanent.counters ?? {})
      .filter(([, count]) => count > 0)
      .map(([kind, count]) => `${count} ${kind.replaceAll('_', ' ')} counter${count > 1 ? 's' : ''}`),
    permanent.attached_to !== undefined ? `attached to ${named(permanent.attached_to)}` : null,
    ...table
      .filter(other => other.attached_to === permanent.id)
      .map(other => `${named(other.id)} attached`),
    permanent.creature_types !== undefined
      ? `creature types set to ${permanent.creature_types.join(' ') || 'none'}`
      : null,
    permanent.is_granted_land ? 'land through an effect' : null,
    permanent.granted_firebending
      ? `firebending ${permanent.granted_firebending} until end of turn`
      : null,
    permanent.exhausted_abilities ? `exhaust abilities used ${permanent.exhausted_abilities}` : null,
    permanent.once_each_turn_used
      ? `once-each-turn abilities used this turn ${permanent.once_each_turn_used}`
      : null,
    permanent.attacking ? 'attacking' : null,
    permanent.blocking !== undefined ? `blocking ${named(permanent.blocking)}` : null,
    permanent.abilities_lost ? 'has lost all abilities' : null,
    permanent.used_abilities?.length
      ? `used-up abilities ${permanent.used_abilities.map(index => index + 1).join(', ')}`
      : null,
    ...(permanent.effects ?? []).map(effectNote),
  ].filter((note): note is string => note !== null);
}
