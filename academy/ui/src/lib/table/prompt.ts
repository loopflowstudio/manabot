/**
 * The prompt's one line: what to do now, in plain words.
 *
 * `promptLine(frame, view)` names the decision and the input that answers it
 * ("Choose blockers", "Pay {2}{U}: Space pays automatically"). The engine's
 * own instruction stays available behind the prompt's details; this line adds
 * no rules and chooses nothing.
 */
import type { ExperienceFrame } from '$lib/types';
import { stackName, type TableView } from './view';

function anyCards(view: TableView): boolean {
  const permanents = [view.you, view.opponent].flatMap(side => [...side.creatures, ...side.support])
    .flatMap(pile => pile.items).flatMap(item => [item, ...item.attached]);
  return view.hand.some(item => item.offers.length) || permanents.some(item => item.offers.length)
    || view.stack.some(item => item.offers.length) || view.you.offers.length + view.opponent.offers.length > 0;
}

function yourMain(frame: ExperienceFrame): boolean {
  const { turn, agent } = frame.projection;
  return turn.active_player_id === agent.id && /MAIN/i.test(`${turn.phase} ${turn.step}`);
}

/** The creature a combat prompt asks about, when it is exactly one of yours. */
function askedCreature(frame: ExperienceFrame, view: TableView): string | null {
  const asked = frame.projection.agent.battlefield.filter(permanent => view.asked.has(permanent.id));
  return asked.length === 1 ? asked[0].name ?? null : null;
}

export function promptLine(frame: ExperienceFrame, view: TableView): string {
  const context = frame.projection.decision_context;
  const creature = askedCreature(frame, view);
  const source = context?.source?.label ?? null;
  const top = view.stack[0] ? stackName(view.stack[0].entry) : null;
  switch (frame.action_space) {
    case 'PRIORITY':
      if (top) return `${top} is on the stack. Respond, or Space to let it resolve`;
      if (!anyCards(view)) return 'Nothing to play. Space passes';
      return yourMain(frame) ? 'Play a land or cast a spell, or Space to pass' : 'Cast or activate now, or Space to pass';
    case 'PAY_OR_NOT':
      return context?.mana_cost
        ? `Pay ${context.mana_cost}${source ? ` for ${source}` : ''}. ${view.primary ? 'Space pays automatically, or click' : 'Click'} lands to tap them`
        : `Pay the cost${source ? ` of ${source}` : ''}, or decline`;
    case 'CHOOSE_TARGET':
      return `Choose a target${source ? ` for ${source}` : ''}`;
    case 'DECLARE_ATTACKER':
      return creature ? `Attack with ${creature}?` : 'Choose attackers';
    case 'DECLARE_BLOCKER':
      return creature ? `Block with ${creature}: click an attacker, or don't block` : 'Choose blockers: click an attacker to block it';
    case 'LEARN':
      return 'Learn: take a Lesson, discard and draw, or skip';
    case 'ORDER_TRIGGERS':
      return 'Choose which trigger goes on the stack next';
    case 'DISCARD':
      return 'Choose a card to discard';
    case 'SCRY':
      return 'Scry: keep each card on top, or put it on the bottom';
    case 'LOOK_AND_SELECT':
      return `Choose from the revealed cards${source ? ` (${source})` : ''}`;
    case 'MODAL':
      return `Choose a mode${source ? ` for ${source}` : ''}`;
    case 'LEGEND_RULE':
      return 'Choose which legendary permanent to keep';
    default:
      return frame.prompt?.title ?? '';
  }
}
