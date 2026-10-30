import { copyLabel } from './decision-labels';
import type { PermanentState } from './types';

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
    permanent.granted_firebending ? `firebending ${permanent.granted_firebending}` : null,
    permanent.attacking ? 'attacking' : null,
    permanent.blocking !== undefined ? `blocking ${named(permanent.blocking)}` : null,
    permanent.abilities_lost ? 'has lost all abilities' : null,
    permanent.used_abilities?.length
      ? `used-up abilities ${permanent.used_abilities.map(index => index + 1).join(', ')}`
      : null,
  ].filter((note): note is string => note !== null);
}
