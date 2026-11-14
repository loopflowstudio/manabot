/** Uncommitted live input, bound to the exact authority frame that offered it.
 * Navigation subjects are not proof of a unique action. Only a unique hand
 * play/cast source may commit by drag; all other choices keep explicit controls.
 */
import { offerFocus } from './offer-map';
import type { ExperienceFrame, InteractionOffer } from './types';

export type TableMode = 'play' | 'inspect';
export const TABLE_MODE_KEY = 'etude.gui.table-mode';
export interface PlaySelection {
  match: string;
  revision: number;
  prompt: number;
  hash: string;
  object: number;
  offers: number[];
}
export interface HandDrag {
  selection: PlaySelection;
  pointer: number;
  x: number;
  y: number;
  moved: boolean;
}

export function selectForPlay(frame: ExperienceFrame, object: number): PlaySelection | null {
  if (!frame.prompt) return null;
  return { match: frame.match_id, revision: frame.revision, prompt: frame.prompt.id,
    hash: frame.frame_hash, object, offers: frame.offers.filter(offer => offerFocus(offer).includes(object)).map(offer => offer.id) };
}

export function selectionMatches(selection: PlaySelection, frame: ExperienceFrame | null): boolean {
  return frame !== null && selection.match === frame.match_id && selection.revision === frame.revision
    && selection.prompt === frame.prompt?.id && selection.hash === frame.frame_hash;
}

export function selectedOffers(selection: PlaySelection | null, frame: ExperienceFrame | null): InteractionOffer[] {
  if (!selection || !frame || !selectionMatches(selection, frame)) return [];
  return frame.offers.filter(offer => selection.offers.includes(offer.id) && offerFocus(offer).includes(selection.object));
}

export function dragOffer(selection: PlaySelection, frame: ExperienceFrame): InteractionOffer | null {
  if (!frame.projection.agent.hand.some(card => card.id === selection.object)) return null;
  const offers = selectedOffers(selection, frame);
  if (offers.length !== 1) return null;
  const offer = offers[0];
  if (offer.verb !== 'cast' && offer.verb !== 'play_land') return null;
  // Published source identity, not a coincidentally matching target/focus.
  const source = offer.subjects?.find(subject => subject.role === 'source')?.object.subject ?? offer.source;
  return source?.kind === 'object' && source.id.entity === selection.object ? offer : null;
}

export function dragMoved(drag: HandDrag, x: number, y: number): boolean {
  return drag.moved || Math.hypot(x - drag.x, y - drag.y) >= 8;
}

export function loadTableMode(): TableMode {
  try { return localStorage.getItem(TABLE_MODE_KEY) === 'play' ? 'play' : 'inspect'; }
  catch { return 'inspect'; }
}
export function saveTableMode(mode: TableMode): void {
  try { localStorage.setItem(TABLE_MODE_KEY, mode); } catch { /* Preference remains usable without storage. */ }
}
