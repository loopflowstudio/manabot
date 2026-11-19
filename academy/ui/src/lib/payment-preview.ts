/** A single read-only query, cached only within the exact live decision. */
import type { ExperienceFrame, PlayerState, SubjectRef } from './types';

export const PAYMENT_PREVIEW_CONTEXT = Symbol('payment-preview');
export interface PreviewPool {
  amounts: number[];
  classes: NonNullable<PlayerState['pooled_mana_classes']>;
}
export interface PaymentPreview {
  state: 'exact' | 'needs_choice' | 'not_applicable' | 'unavailable';
  explanation: string;
  cost: string | null;
  /** What objects have already paid toward the cost, in the order paid. */
  contributions: string[];
  sources: { source: SubjectRef; label: string; output: PreviewPool }[];
  pool_after_production: PreviewPool | null;
  pool_after_spending: PreviewPool | null;
}
export interface PreviewAddress {
  match_id: string; revision: number; prompt_id: number; frame_hash: string;
  offer_id: number;
}
export interface PreviewResponse extends PreviewAddress { viewer: number; preview: PaymentPreview }
export type PreviewFetcher = (address: PreviewAddress, signal: AbortSignal) => Promise<PreviewResponse>;
export interface PreviewState { offer: number; label: string; loading: boolean; error: boolean; preview: PaymentPreview | null }

function frameKey(frame: ExperienceFrame): string {
  return `${frame.match_id}:${frame.revision}:${frame.prompt?.id}:${frame.frame_hash}:${frame.prompt?.actor}`;
}
export class PaymentPreviewController {
  private frame = '';
  private active: AbortController | null = null;
  private cache = new Map<number, PaymentPreview>();
  state: PreviewState | null = null;
  constructor(private fetcher: PreviewFetcher, private changed: (state: PreviewState | null) => void) {}
  clear(): void {
    this.active?.abort();
    this.active = null;
    this.state = null;
    this.changed(null);
  }
  invalidate(): void { this.clear(); this.frame = ''; this.cache.clear(); }
  async select(frame: ExperienceFrame, offer: number, label: string): Promise<void> {
    if (!frame.prompt || !frame.offers.some(candidate => candidate.id === offer)) return;
    const key = frameKey(frame);
    if (key !== this.frame) { this.invalidate(); this.frame = key; }
    if (this.state?.offer === offer && !this.state.error) return;
    this.clear();
    const cached = this.cache.get(offer);
    this.state = { offer, label, loading: !cached, error: false, preview: cached ?? null };
    this.changed(this.state);
    if (cached) return;
    const address: PreviewAddress = { match_id: frame.match_id, revision: frame.revision,
      prompt_id: frame.prompt.id, frame_hash: frame.frame_hash, offer_id: offer };
    const controller = new AbortController();
    this.active = controller;
    try {
      const response = await this.fetcher(address, controller.signal);
      // Every selection change and authority invalidation aborts this signal,
      // even when the fetcher has already received a response.
      if (controller.signal.aborted) return;
      if (Object.entries(address).some(([name, value]) => response[name as keyof PreviewAddress] !== value)
        || response.viewer !== frame.prompt.actor) throw new Error('Preview authority changed');
      this.cache.set(offer, response.preview);
      this.state = { offer, label, loading: false, error: false, preview: response.preview };
    } catch {
      if (controller.signal.aborted) return;
      this.state = { offer, label, loading: false, error: true, preview: null };
    }
    this.changed(this.state);
  }
}
