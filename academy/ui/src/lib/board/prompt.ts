/**
 * The prompt's one line: what to do now, in plain words.
 *
 * `promptLine(frame, view, keys)` names the decision ("Choose blockers", "Pay {2}{U}");
 * `keys` false (a recording, where Space plays the transport) leaves out the
 * live keyboard's words.
 * `promptHint(frame, view)` describes how to answer a mana payment. The engine's
 * own instruction stays available behind the prompt's details; this line adds
 * no rules and chooses nothing.
 */
import type { ExperienceFrame } from '$lib/types';
import type { PaymentPreview } from '$lib/payment-preview';
import { payLabel, paymentPlan } from './payment';
import { stackName, type BoardView } from './view';

function anyCards(view: BoardView): boolean {
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
function askedCreature(frame: ExperienceFrame, view: BoardView): string | null {
  const asked = frame.projection.agent.battlefield.filter(permanent => view.asked.has(permanent.id));
  return asked.length === 1 ? asked[0].name ?? null : null;
}

/** What the decision asks, in one stable line: hovering a choice never rewrites it. */
export function promptLine(frame: ExperienceFrame, view: BoardView, keys = true): string {
  const context = frame.projection.decision_context;
  const creature = askedCreature(frame, view);
  const source = context?.source?.label ?? null;
  const top = view.stack[0] ? stackName(view.stack[0].entry) : null;
  switch (frame.action_space) {
    case 'PRIORITY':
      if (top) return `${top} is on the stack. Respond, or ${keys ? 'Space to ' : ''}let it resolve`;
      if (!anyCards(view)) return keys ? 'Nothing to play. Space passes' : 'Nothing to play';
      return `${yourMain(frame) ? 'Play a land or cast a spell' : 'Cast or activate now'}, or ${keys ? 'Space to ' : ''}pass`;
    case 'PAY_OR_NOT':
      return context?.mana_cost
        ? `Pay ${context.mana_cost}${source ? ` for ${source}` : ''}`
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
      // Mid-payment, a modal choice is what a tap pays, not a mode of the spell.
      if (context?.stage === 'payment') return `Choose what the tap pays${source ? ` for ${source}` : ''}`;
      if (context?.stage === 'assign_combat_damage') return 'Assign combat damage';
      if (context?.stage === 'order_replacement') return 'Choose which replacement applies next';
      if (context?.stage === 'target_shares') return `Divide among targets${source ? ` for ${source}` : ''}`;
      if (context?.stage === 'entry_choice') return `Choose as ${source ?? 'the permanent'} enters`;
      if (context?.stage === 'order_entry_choices') return 'Choose which entering permanent to decide next';
      if (context?.stage === 'partition_piles') return `Put each card into the face-up or face-down pile${source ? ` (${source})` : ''}`;
      if (context?.stage === 'choose_pile') return `Choose which pile goes to hand${source ? ` (${source})` : ''}`;
      return `Choose a mode${source ? ` for ${source}` : ''}`;
    case 'LEGEND_RULE':
      return 'Choose which legendary permanent to keep';
    default:
      return frame.prompt?.title ?? '';
  }
}

/** The payment instructions hold steady as previews arrive and change. */
export function promptHint(frame: ExperienceFrame, view: BoardView): string | null {
  if (frame.action_space !== 'PAY_OR_NOT' || !frame.projection.decision_context?.mana_cost) return null;
  if (!view.primary) return 'Click lands to tap them yourself';
  return 'Space pays automatically, or click lands to tap them yourself';
}

/**
 * What is committed to the cost being paid, in the engine's words and order:
 * a chosen sacrifice has not left yet.
 */
export function committed(frame: ExperienceFrame): string[] {
  return (frame.projection.decision_context?.paid ?? []).map(entry => entry.phrase);
}

/** Where passing priority on an empty stack leads: the next step, in words. */
const NEXT_STEP: Record<string, string> = {
  BEGINNING_UNTAP: 'To draw', BEGINNING_UPKEEP: 'To draw', BEGINNING_DRAW: 'To main',
  PRECOMBAT_MAIN_STEP: 'To combat', COMBAT_BEGIN: 'To attackers', COMBAT_DECLARE_ATTACKERS: 'To blockers',
  DECLARE_ATTACKERS_STEP: 'To blockers', COMBAT_DECLARE_BLOCKERS: 'To damage', DECLARE_BLOCKERS_STEP: 'To damage',
  COMBAT_FIRST_STRIKE_DAMAGE: 'To regular damage', COMBAT_DAMAGE: 'To end of combat', COMBAT_END: 'To main 2',
  POSTCOMBAT_MAIN_STEP: 'To end step', ENDING_END: 'To cleanup', END_STEP: 'To cleanup', ENDING_CLEANUP: 'To next turn',
};

/** The most characters the primary button shows on its one line, beside its key. */
export const PRIMARY_CHARS = 24;

export interface PrimaryLabel {
  /** What pressing it does next, shortly: "Resolve Lightning Bolt", "To combat", "Pay {1}{U}". */
  text: string;
  /** The offer it commits and what follows, for the accessible name. */
  description: string;
}

/**
 * An offer's words on the board. The payment offer hands the choice of lands
 * to the automatic payer, so it reads "Auto pay"; the recorded label stays
 * "Pay the cost".
 */
export function offerText(offer: { verb?: string; label: string }): string {
  return offer.verb === 'pay' ? offer.label.replace(/^Pay the cost/, 'Auto pay') : offer.label;
}

/**
 * The primary button's words: what pressing it does next rather than the
 * offer's verb. Passing with a spell on the stack lets it resolve (if the
 * other player passes too); on an empty stack it moves to the next step. The
 * offer committed is unchanged and named in the description.
 */
export function primaryLabel(frame: ExperienceFrame, view: BoardView, preview: PaymentPreview | null = null): PrimaryLabel | null {
  const primary = view.primary;
  if (!primary) return null;
  const offer = primary.offer.label;
  if (primary.offer.verb === 'pay') {
    // The plan is the automatic payer's, read for this offer; without one the button keeps the cost.
    const full = payLabel(frame.projection.decision_context?.mana_cost ?? null, paymentPlan(preview, frame), preview);
    // The button says the price on one line; the lands it taps are drawn on
    // the board and named in its accessible name.
    if (full) return { text: full.split(' · ')[0], description: `${offer}: ${full.slice('Auto pay '.length)}` };
    return { text: offerText(primary.offer), description: offerText(primary.offer) };
  }
  const top = view.stack[0] ? stackName(view.stack[0].entry) : null;
  // A card name is never cut: when it does not fit beside the verb, the
  // button says the verb and the prompt line above names what resolves.
  if (top) return { text: `Resolve ${top}`.length <= PRIMARY_CHARS ? `Resolve ${top}` : 'Resolve', description: `${offer}: let ${top} resolve` };
  const { turn, agent, opponent } = frame.projection;
  let next = NEXT_STEP[turn.step];
  if (turn.step === 'COMBAT_DECLARE_ATTACKERS' || turn.step === 'DECLARE_ATTACKERS_STEP') {
    const attacking = [agent, opponent].some(player => player.battlefield.some(permanent => permanent.attacking));
    if (!attacking) next = 'To end of combat';
  }
  return next ? { text: next, description: `${offer}: ${next.toLowerCase()}` } : { text: offer, description: offer };
}
