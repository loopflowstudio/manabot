/**
 * The automatic payer's plan, as the board draws it.
 *
 * The native payment preview (`/payment-preview`, `$lib/payment-preview`)
 * answers for one offer: which sources **Pay the cost** would tap and what
 * each produces. `paymentPlan(preview, frame)` turns an exact answer into
 * marks on the lands it names ("would tap, for {U}") and the words the
 * primary button reads ("Pay {1}{U} · Island ×2"). It works nothing out:
 * a preview that is not exact marks nothing, and the button keeps the cost.
 */
import type { PaymentPreview, PreviewPool } from '$lib/payment-preview';
import type { ExperienceFrame, InteractionOffer } from '$lib/types';

const SYMBOLS = ['W', 'U', 'B', 'R', 'G', 'C'];

/** Offers whose preview the authority answers: casting, activating, targeting and paying. */
export const PREVIEW_TYPES = new Set(['PRIORITY_CAST_SPELL', 'PRIORITY_ACTIVATE_ABILITY', 'CHOOSE_TARGET', 'PAY_COST', 'TAP_FOR_COST']);

export function previewable(offer: InteractionOffer): boolean {
  return PREVIEW_TYPES.has(offer.action_type);
}

/** A pool as mana symbols, "{U}{U}{C}"; empty for an empty pool. */
export function poolSymbols(amounts: readonly number[]): string {
  return amounts.flatMap((amount, index) => Array.from({ length: Math.max(0, amount) }, () => `{${SYMBOLS[index] ?? 'C'}}`)).join('');
}

export interface PlannedSource {
  /** One-based tapping order. */
  order: number;
  /** What tapping it adds, as symbols. */
  output: string;
}

export interface PaymentPlan {
  /** Permanent id → its place in the plan. */
  sources: Map<number, PlannedSource>;
  /** The cost the plan pays, as the preview states it. */
  cost: string | null;
  /** The sources' names in tapping order, for the button. */
  names: string[];
}

function emptyPool(pool: PreviewPool | null): boolean {
  return !pool || !pool.amounts.some(Boolean);
}

/** The plan to draw for an exact preview, or null when there is nothing to draw. */
export function paymentPlan(preview: PaymentPreview | null, frame: ExperienceFrame): PaymentPlan | null {
  if (!preview || preview.state !== 'exact') return null;
  const names = new Map([...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield]
    .map(permanent => [permanent.id, permanent.name ?? 'permanent']));
  const sources = new Map<number, PlannedSource>();
  const listed: string[] = [];
  preview.sources.forEach((source, index) => {
    const id = source.source.kind === 'object' ? source.source.id.entity : null;
    if (id !== null) sources.set(id, { order: index + 1, output: poolSymbols(source.output.amounts) });
    listed.push((id !== null ? names.get(id) : null) ?? source.label);
  });
  return { sources, cost: preview.cost, names: listed };
}

/** "Island ×2, Mountain": names in order of first appearance, each with its count. */
function counted(names: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name)).join(', ');
}

/** The primary button for a payment: auto pay the cost, then what pays it. */
export function payLabel(cost: string | null, plan: PaymentPlan | null, preview: PaymentPreview | null): string | null {
  const price = cost ?? plan?.cost ?? null;
  if (!price) return null;
  if (!plan) return `Auto pay ${price}`;
  if (plan.names.length) return `Auto pay ${price} · ${counted(plan.names)}`;
  return emptyPool(preview?.pool_after_production ?? null) ? `Auto pay ${price}` : `Auto pay ${price} · floating mana`;
}
