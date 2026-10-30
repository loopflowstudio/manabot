/** Player-facing names for exact subjects; native IDs remain internal bindings. */
import type { DecisionObjectView, Observation } from './types';

type NamedCard = { id: number; name: string | null };

export function copyLabel(cards: NamedCard[], id: number): string {
  const card = cards.find(card => card.id === id);
  if (!card) return '';
  const copies = cards.filter(candidate => candidate.name === card.name);
  return copies.length > 1 ? `copy ${copies.findIndex(copy => copy.id === id) + 1} of ${copies.length}` : '';
}

export function visibleCardLabel(observation: Observation, id: number): string | null {
  for (const [index, player] of [observation.agent, observation.opponent].entries()) {
    for (const [zone, cards] of [
      ['hand', player.hand], ['battlefield', player.battlefield],
      ['graveyard', player.graveyard], ['exile', player.exile],
      ['stack', player.stack], ['private inspection', player.inspected ?? []],
    ] as const) {
      const card = cards.find(card => card.id === id);
      if (card) return [card.name ?? 'Unknown card', `${index === 0 ? 'your' : "opponent's"} ${zone}`, copyLabel(cards, id)].filter(Boolean).join(' · ');
    }
  }
  return null;
}

export function subjectLabel(observation: Observation, object: DecisionObjectView): string {
  return object.subject.kind === 'player' ? `Seat ${object.subject.id + 1}` :
    visibleCardLabel(observation, object.display_id) ?? object.label;
}

export function stageLabel(stage: string): string {
  return ({
    order_triggers: 'Choose the next ability to stack', trigger_targets: 'Choose trigger targets',
    priority: 'Choose an action', targets: 'Choose a target', mode: 'Choose a spell mode',
    additional_cost: 'Choose an additional cost', payment: 'Pay the announced cost',
    mana_color: 'Choose mana to add', declare_attackers: 'Declare attackers',
    declare_blockers: 'Declare blockers', resolution_sacrifice: 'Sacrifice during resolution',
    private_search: 'Search your library', learn: 'Learn', private_inspection: 'Inspect cards',
    scry: 'Scry', surveil: 'Surveil', resolution_payment: 'Pay during resolution', resolution_mode: 'Choose an effect',
    discard: 'Choose a discard', select_milled: 'Choose a milled card', legend_rule: 'Choose a legend to keep',
  } as Record<string, string>)[stage] ?? 'Pending decision';
}
