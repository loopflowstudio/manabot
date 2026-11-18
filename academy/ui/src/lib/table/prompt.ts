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

/** Where passing priority on an empty stack leads: the next step, in words. */
const NEXT_STEP: Record<string, string> = {
  BEGINNING_UNTAP: 'To draw', BEGINNING_UPKEEP: 'To draw', BEGINNING_DRAW: 'To main',
  PRECOMBAT_MAIN_STEP: 'To combat', COMBAT_BEGIN: 'To attackers', COMBAT_DECLARE_ATTACKERS: 'To blockers',
  DECLARE_ATTACKERS_STEP: 'To blockers', COMBAT_DECLARE_BLOCKERS: 'To damage', DECLARE_BLOCKERS_STEP: 'To damage',
  COMBAT_FIRST_STRIKE_DAMAGE: 'To damage', COMBAT_DAMAGE: 'To end of combat', COMBAT_END: 'To main 2',
  POSTCOMBAT_MAIN_STEP: 'To end step', ENDING_END: 'To cleanup', END_STEP: 'To cleanup', ENDING_CLEANUP: 'To next turn',
};

export interface PrimaryLabel {
  /** What pressing it does next, shortly: "Resolve Lightning Bolt", "To combat", "Pay {1}{U}". */
  text: string;
  /** The offer it commits and what follows, for the accessible name. */
  description: string;
}

/**
 * The primary button's words: what pressing it does next rather than the
 * offer's verb. Passing with a spell on the stack lets it resolve (if the
 * other player passes too); on an empty stack it moves to the next step. The
 * offer committed is unchanged and named in the description.
 */
export function primaryLabel(frame: ExperienceFrame, view: TableView): PrimaryLabel | null {
  const primary = view.primary;
  if (!primary) return null;
  const offer = primary.offer.label;
  if (primary.offer.verb === 'pay') {
    const cost = frame.projection.decision_context?.mana_cost;
    return cost ? { text: `Pay ${cost}`, description: `${offer}: ${cost}` } : { text: offer, description: offer };
  }
  const top = view.stack[0] ? stackName(view.stack[0].entry) : null;
  if (top) return { text: `Resolve ${top}`, description: `${offer}: let ${top} resolve` };
  const { turn } = frame.projection;
  const attackers = [frame.projection.agent, frame.projection.opponent].some(player => player.battlefield.some(permanent => permanent.attacking));
  const declaring = turn.step === 'COMBAT_DECLARE_ATTACKERS' || turn.step === 'DECLARE_ATTACKERS_STEP';
  const next = declaring && !attackers ? 'To end of combat' : NEXT_STEP[turn.step];
  return next ? { text: next, description: `${offer}: ${next.toLowerCase()}` } : { text: offer, description: offer };
}
