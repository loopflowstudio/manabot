/** Privacy checks shared by imported replay, Study and advice readers. */
import type { Observation } from './types';

export function exposesOtherPrivateContext(observation: Observation): boolean {
  const context = observation.decision_context;
  return context !== undefined && context.actor !== observation.agent.player_index && (
    context.source !== null || context.targets.length > 0 || context.reserved_sacrifice !== null ||
    context.mana_cost !== null || context.chosen_mode !== null
  );
}
