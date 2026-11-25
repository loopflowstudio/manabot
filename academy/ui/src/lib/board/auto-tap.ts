/**
 * Arena-style Auto-tap and Full Control (`Ctrl` / `Shift+Ctrl`) over the shared
 * decision surface.
 *
 * The engine stays in Manual mana payment (`ManaPayment::Manual`). When
 * Auto-tap is on and Full Control is off, Academy submits the legal `pay`
 * (`Auto pay`) offer at mandatory pure-mana `PAY_OR_NOT` prompts and records
 * it as automation (`actor: 'auto'`). Optional costs (`decline` offered) and
 * non-mana payment choices (convoke, improvise, delve, sacrifice, discard,
 * life) are never auto-submitted.
 */
import type { ExperienceFrame } from '$lib/types';
import type { BoardView, PlacedOffer } from './view';

export const AUTO_TAP_STORAGE_KEY = 'academy.auto-tap';

export type FullControlMode = 'off' | 'once' | 'hold';

const PURE_MANA_ACTION_TYPES = new Set([
  'PAY_COST',
  'TAP_FOR_COST',
  'TAP_FOR_MANA',
  'ACTIVATE_MANA_ABILITY',
  'PRIORITY_ACTIVATE_ABILITY',
]);

export function loadAutoTap(): boolean {
  if (typeof localStorage === 'undefined') return true;
  try {
    return localStorage.getItem(AUTO_TAP_STORAGE_KEY) !== 'off';
  } catch {
    return true;
  }
}

export function saveAutoTap(enabled: boolean): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(AUTO_TAP_STORAGE_KEY, enabled ? 'on' : 'off');
  } catch {
    /* Private browsing / quota: keep in-memory state. */
  }
}

/** Arena keyboard transitions for releasing bare `Ctrl` or `Shift+Ctrl`. */
export function nextFullControl(current: FullControlMode, shiftHeld: boolean): FullControlMode {
  if (shiftHeld) return current === 'hold' ? 'off' : 'hold';
  return current === 'off' ? 'once' : 'off';
}

/**
 * Returns the primary `pay` offer to auto-submit, or `null` when the player
 * should decide manually.
 */
export function autoTapOffer(
  frame: ExperienceFrame,
  view: BoardView,
  options: { autoTap: boolean; fullControl: FullControlMode },
): PlacedOffer | null {
  if (!options.autoTap || options.fullControl !== 'off') return null;
  if (frame.action_space !== 'PAY_OR_NOT') return null;
  if (!frame.projection.decision_context?.mana_cost) return null;
  if (!view.primary || view.primary.offer.verb !== 'pay') return null;
  if (frame.offers.some(offer => !PURE_MANA_ACTION_TYPES.has(offer.action_type))) return null;
  if (frame.offers.some(offer => /\bdecline\b/i.test(offer.label))) return null;
  return view.primary;
}

/** Move-log automation caption so auto-tap stays separate from deliberate moves. */
export function autoTapNote(frame: ExperienceFrame): string {
  const ctx = frame.projection.decision_context;
  const cost = ctx?.mana_cost ?? 'the cost';
  const source = ctx?.source?.label;
  return source ? `Auto-tapped ${cost} for ${source}` : `Auto-tapped ${cost}`;
}
