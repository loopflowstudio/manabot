<script lang="ts">
  /**
   * One card on the table: a permanent, a card in hand, a spell on the stack
   * or a tray candidate. It is a button whenever the viewer may inspect it;
   * its outline says a legal offer acts on it, and its accessible name says
   * which.
   *
   * Both frames are a printed card's proportions (63:88). `portrait` (hand,
   * tray) is art over a bar with the name and type line, cost pips on the
   * art. `field` (battlefield) puts the name bar on top, wrapping to a second
   * line rather than cutting the name, with the art below and no cost (the
   * hand and inspect show it). A tapped permanent lies on its side: its
   * footprint and art turn a quarter, while the name bar stays across the
   * top so it still reads. Counters, pile counts, keywords and power and
   * toughness sit on the art, upright. Rules text lives in the inspect view.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import { cardFace } from '$lib/card-faces.svelte';
  import type { CardState, PermanentState } from '$lib/types';
  import { getContext } from 'svelte';
  import { keywordsOf } from './inspect';
  import { cardEvidence, EVIDENCE, formatProbability, type OfferEvidence } from './replay';
  import type { PlannedSource } from './payment';
  import Pips from './Pips.svelte';
  import { footprint, type PlacedOffer } from './view';

  interface Props {
    name: string;
    /** Upright width, px; a tapped permanent's footprint is this card on its side. */
    width: number;
    shape?: 'portrait' | 'field';
    permanent?: PermanentState | null;
    card?: CardState | null;
    offers?: PlacedOffer[];
    /** Offers may be committed now. */
    live?: boolean;
    asked?: boolean;
    /** The side of the centre line this card sits on; attacks lean toward it. */
    toward?: 'up' | 'down' | null;
    /** The name of the attacker this permanent blocks; the line on the table draws it. */
    blocks?: string | null;
    /** Whose plate a card new to the table travels from. */
    side?: 'you' | 'opponent';
    /** It attacked, blocked, was dealt damage or was targeted since your last move. */
    touched?: boolean;
    /** The opponent's move being read out is about this card. */
    beat?: boolean;
    /** Damage it was dealt since your last move, shown as it lands. */
    dealt?: number;
    /** Its place in the automatic payer's plan: "would tap, for {U}". */
    planned?: PlannedSource | null;
    /** Copies in this card's pile, shown on the top copy. */
    count?: number | null;
    /** Its place in a combat draft: it may join (`can`), it is drafted (`in`), or it is the blocker being placed (`armed`). */
    draft?: 'can' | 'in' | 'armed' | null;
    /** What pressing it does in a combat draft, for its accessible name. */
    note?: string | null;
    onactivate?: (element: HTMLElement) => void;
    onpreview?: (element: HTMLElement | null) => void;
  }

  let {
    name, width, shape = 'portrait', permanent = null, card = null, offers = [], live = false, asked = false,
    toward = null, blocks = null, side = 'you', touched = false, beat = false, dealt = 0, planned = null, count = null, draft = null, note = null, onactivate, onpreview,
  }: Props = $props();

  // At a recorded choice: the policy's mass on this card's offers, and whether one was played.
  const recordedPolicy = getContext<(() => OfferEvidence | null) | undefined>(EVIDENCE);
  const recorded = $derived(cardEvidence(recordedPolicy?.() ?? null, offers.map(placed => placed.offer.id)));

  const field = $derived(shape === 'field');
  const tapped = $derived(permanent?.tapped ?? false);
  const turned = $derived(field && tapped);
  const upright = $derived(footprint(width, true));
  // A turned card's box is the upright card on its side.
  const boxWidth = $derived(turned ? upright : width);
  const boxHeight = $derived(turned ? width : upright);
  const creature = $derived(permanent ? permanent.types?.is_creature ?? permanent.power !== null : card?.types.is_creature ?? false);
  const power = $derived(permanent?.power ?? card?.power ?? null);
  const toughness = $derived(permanent?.toughness ?? card?.toughness ?? null);
  const face = $derived(cardFace(permanent?.copy_of?.name ?? name));
  const keywords = $derived(permanent ? keywordsOf(permanent, face) : (face?.keywords ?? []).map(keyword => ({ keyword, source: null })));
  const counters = $derived(Object.entries({
    ...(permanent?.plus1_counters ? { '+1/+1': permanent.plus1_counters } : {}),
    ...(permanent?.counters ?? {}),
  }).filter(([, amount]) => amount > 0));

  // Type sizes follow the frame, within a readable floor and a quiet ceiling.
  // Battlefield names are 12 px at every size: a crowded row wraps them.
  const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));
  const nameSize = $derived(field ? 12 : clamp(upright * 0.072, 11, 15));
  const typeSize = $derived(clamp(upright * 0.05, 9, 11));
  const showType = $derived(!field && upright >= 110);
  const pip = $derived(clamp(upright * 0.07, 10, 14));
  const showCost = $derived(!field && width >= 58);
  const plate = $derived(clamp(upright * 0.085, 10, 14));
  const showKeywords = $derived(field && width >= 64 && keywords.length > 0);
  const keywordRoom = $derived(boxWidth >= 120 ? 3 : 1);

  const url = $derived(artUrl(name));
  let loaded = $state<string | null>(null);
  $effect(() => {
    const requested = url;
    let active = true;
    void loadArt(requested, name).then(ok => { if (active && ok) loaded = requested; });
    return () => { active = false; };
  });
  const art = $derived(loaded === url ? url : null);

  const facts = $derived([
    tapped ? 'tapped' : '',
    permanent?.summoning_sick && creature ? 'summoning sick' : '',
    permanent?.attacking ? 'attacking' : '',
    permanent?.blocking !== undefined ? `blocking ${blocks ?? 'an attacker'}` : '',
    permanent?.damage ? `${permanent.damage} damage` : '',
    ...counters.map(([kind, amount]) => `${amount} ${kind} counter${amount === 1 ? '' : 's'}`),
    permanent?.is_token ? 'token' : '',
    dealt ? `dealt ${dealt} since your last move` : '',
    planned ? `would tap for ${planned.output || 'mana'} (payment plan, source ${planned.order})` : '',
    recorded?.p != null ? `recorded policy ${formatProbability(recorded.p)}` : '',
    recorded?.chosen ? 'chosen' : '',
  ].filter(Boolean).join(', '));
  const label = $derived([
    `${name}${creature && power !== null ? ` ${power}/${toughness}` : ''}`,
    facts,
    note ?? '',
    offers.length === 1 ? `${offers[0].offer.label} (${offers[0].number})`
      : offers.length > 1 ? `${offers.length} choices: ${offers.map(placed => placed.offer.label).join('; ')}` : '',
  ].filter(Boolean).join(' — '));
</script>

<div
  class="slot"
  class:lean-up={permanent?.attacking && toward === 'up'}
  class:lean-down={permanent?.attacking && toward === 'down'}
  style:width={`${boxWidth}px`}
  style:height={`${boxHeight}px`}
>
  <button
    type="button"
    class="tcard"
    class:field
    class:turned
    class:legal={live && (offers.length > 0 || draft === 'can')}
    class:drafted={draft === 'in'}
    class:armed={draft === 'armed'}
    class:offered={!live && recorded !== null}
    class:chosen={recorded?.chosen}
    class:asked
    class:attacking={permanent?.attacking}
    class:touched
    class:beat
    class:token={permanent?.is_token}
    class:planned={planned !== null}
    data-testid="table-card"
    data-travel-id={permanent ? `perm:${permanent.id}` : card ? `card:${card.id}` : undefined}
    data-travel-name={name}
    data-travel-side={side}
    data-object-id={permanent?.id ?? card?.id}
    data-offer-count={offers.length}
    data-draft={draft ?? undefined}
    data-payment-source={planned?.order}
    data-card-name={name}
    aria-label={label}
    style:width={`${boxWidth}px`}
    style:height={`${boxHeight}px`}
    onclick={event => onactivate?.(event.currentTarget)}
    onmouseenter={event => onpreview?.(event.currentTarget)}
    onmouseleave={() => onpreview?.(null)}
    onfocus={event => onpreview?.(event.currentTarget)}
    onblur={() => onpreview?.(null)}
  >
    <span class="bar" aria-hidden="true">
      <span class="name" style:font-size={`${nameSize}px`}>{name}</span>
      {#if showType && face}<span class="type" style:font-size={`${typeSize}px`}>{face.type_line}</span>{/if}
    </span>
    <span class="art" aria-hidden="true">
      <span class="picture" style:background-image={art ? `url('${art}')` : undefined}>
        {#if !art}<span class="no-art">{face?.type_line ?? ''}</span>{/if}
      </span>
      {#if counters.length && width >= 56}
        <span class="counters">{counters.map(([kind, amount]) => `${kind === '+1/+1' ? '+1' : kind.replace(/_/g, ' ')} ×${amount}`).join(' · ')}</span>
      {/if}
      <span class="corner">
        {#if showCost && face?.cost}<span class="cost"><Pips cost={face.cost} size={pip} /></span>{/if}
        {#if count}<span class="count">×{count}</span>{/if}
        {#if offers.length > 1 && live}<span class="many">{offers.length}</span>{/if}
      </span>
      {#if showKeywords}
        <span class="keywords">
          {#each keywords.slice(0, keywordRoom) as fact (fact.keyword)}<i class:granted={fact.source !== null}>{fact.keyword.replace(/_/g, ' ')}</i>{/each}
          {#if keywords.length > keywordRoom}<i>+{keywords.length - keywordRoom}</i>{/if}
        </span>
      {/if}
      {#if creature && power !== null}
        <span class="pt" class:hurt={(permanent?.damage ?? 0) > 0}
          class:up={permanent && ((permanent.power ?? 0) > (face?.power ?? Infinity) || (permanent.toughness ?? 0) > (face?.toughness ?? Infinity))}
          style:font-size={`${plate}px`}>{power}/{toughness}{#if permanent?.damage}<small>−{permanent.damage}</small>{/if}</span>
      {/if}
      {#if planned}
        <span class="plan"><b>{planned.order}</b>{#if planned.output}<Pips cost={planned.output} size={Math.max(10, pip)} />{/if}</span>
      {/if}
      {#if recorded && (recorded.p !== null || recorded.chosen)}
        <span class="policy" data-testid="card-policy" data-chosen={recorded.chosen}>{#if recorded.chosen}✓{/if}{recorded.p !== null ? formatProbability(recorded.p) : ''}</span>
      {/if}
      {#if dealt}{#key dealt}<span class="dealt" data-testid="damage-dealt">−{dealt}</span>{/key}{/if}
    </span>
  </button>
</div>

<style>
  .slot { position: relative; flex: none; transition: transform var(--motion-move) var(--motion-move-ease); }
  /* An attacker steps toward the player it attacks by `--lean`, which its row sizes (Field). */
  .slot.lean-up { transform: translateY(calc(-1 * var(--lean, 20%))); }
  .slot.lean-down { transform: translateY(var(--lean, 20%)); }

  .tcard {
    position: relative; display: flex; flex-direction: column-reverse; overflow: hidden; padding: 0; cursor: pointer; text-align: left;
    border-radius: max(4px, 5%); border: 1px solid rgb(20 14 8 / 0.55);
    background: var(--bg-surface); color: var(--text);
    box-shadow: 0 1px 2px rgb(20 14 8 / 0.25);
    transition: transform var(--motion-move) var(--motion-move-ease), box-shadow var(--motion-fast) ease-out;
  }
  .tcard:focus-visible { outline: 2px solid var(--text); outline-offset: 3px; }
  .turned { border-radius: max(4px, 3.5%); }

  /* The name bar: under the art in hand, across the top on the battlefield, upright either way. */
  .bar {
    flex: none; min-width: 0; padding: 3px 5px 4px; display: flex; flex-direction: column; gap: 1px;
    background: var(--bg-surface); border-top: 1px solid var(--border);
  }
  .field { flex-direction: column; }
  .field .bar { padding: 2px 4px 3px; border-top: 0; border-bottom: 1px solid var(--border); }
  .name {
    font-family: var(--font-sans); font-weight: 700; line-height: 1.2; color: var(--text);
    overflow: hidden; overflow-wrap: break-word; hyphens: auto; text-wrap: balance;
    display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2;
  }
  .type { font-family: var(--font-sans); line-height: 1.2; color: var(--text-tertiary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .token .bar { background: color-mix(in srgb, var(--bg-muted) 60%, var(--bg-surface)); }

  /* The art fills what the bar leaves. A turned card's picture is the upright art turned a quarter. */
  .art { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; container-type: size; background: var(--bg-muted); }
  .picture { position: absolute; inset: 0; background: var(--bg-muted) center 30% / cover no-repeat; }
  .turned .picture {
    inset: auto; left: 50%; top: 50%; width: 100cqh; height: 100cqw;
    transform: translate(-50%, -50%) rotate(90deg);
  }
  .no-art { position: absolute; inset: 4px; display: grid; place-items: center; text-align: center; font: italic 10px/1.2 var(--font-serif); color: var(--text-tertiary); overflow: hidden; }

  .corner { position: absolute; top: 3px; right: 3px; display: flex; gap: 3px; align-items: flex-start; }
  .cost { padding: 1px 2px; border-radius: 8px; background: rgb(247 240 224 / 0.82); display: flex; }
  .count {
    padding: 0 5px; border-radius: 8px; font: 700 12px/16px var(--font-sans); font-variant-numeric: tabular-nums;
    color: var(--text); background: color-mix(in srgb, var(--bg-surface) 90%, transparent); border: 1px solid var(--border-strong);
  }
  .many {
    min-width: 17px; height: 17px; border-radius: 9px; padding: 0 4px;
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 700; line-height: 15px; text-align: center;
    color: #f8f1e0; background: var(--legal); border: 1px solid rgb(20 14 8 / 0.5);
  }
  .counters {
    position: absolute; top: 3px; left: 3px; max-width: calc(100% - 6px); padding: 0 4px; border-radius: 4px; text-align: left;
    font-family: var(--font-sans); font-size: 9.5px; font-weight: 700; line-height: 14px;
    color: #f8f1e0; background: rgb(12 9 5 / 0.72); white-space: nowrap; overflow: hidden;
  }

  .tcard.attacking { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 6px 14px rgb(20 14 8 / 0.3); }
  .tcard.touched { box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 55%, transparent), 0 1px 2px rgb(20 14 8 / 0.25); }
  .tcard.legal { box-shadow: 0 0 0 2px var(--legal), 0 0 12px color-mix(in srgb, var(--legal) 45%, transparent); }
  .tcard.legal:hover { box-shadow: 0 0 0 3px var(--legal), 0 0 16px color-mix(in srgb, var(--legal) 60%, transparent); transform: translateY(-2px); }
  /* A recorded choice's legal space, read-only: offered cards are outlined thinly, the played one solidly. */
  .tcard.offered { box-shadow: 0 0 0 1px color-mix(in srgb, var(--legal) 70%, transparent); }
  .tcard.chosen { box-shadow: 0 0 0 2px var(--text), 0 4px 10px rgb(20 14 8 / 0.3); }
  .policy {
    position: absolute; left: 50%; bottom: 3px; transform: translateX(-50%); padding: 0 5px; border-radius: 8px;
    font: 700 10.5px/15px var(--font-sans); font-variant-numeric: tabular-nums; white-space: nowrap;
    color: var(--text); background: var(--bg-surface); border: 1px solid var(--border-strong);
  }
  .policy[data-chosen='true'] { color: var(--bg-surface); background: var(--text); border-color: var(--text); }
  .tcard.beat { box-shadow: 0 0 0 3px var(--accent), 0 0 18px color-mix(in srgb, var(--accent) 55%, transparent); transition: box-shadow 160ms ease-out; }
  /* Drafted into an attack or block: a dashed accent ring, distinct from a declared (solid) attacker. */
  .tcard.drafted { outline: 2px dashed var(--accent); outline-offset: 2px; }
  .tcard.armed { box-shadow: 0 0 0 2px var(--accent), 0 8px 18px rgb(20 14 8 / 0.35); transform: translateY(-8px); outline: 2px dashed var(--accent); outline-offset: 2px; }
  .tcard.asked { box-shadow: 0 0 0 2px var(--accent), 0 8px 18px rgb(20 14 8 / 0.35); transform: translateY(-8px); }

  /* Planned payment: dashed, and what tapping it adds. Distinct from the solid "legal" outline. */
  .tcard.planned { outline: 2px dashed var(--planned); outline-offset: 2px; }
  .plan {
    position: absolute; left: 50%; top: 3px; transform: translateX(-50%);
    display: inline-flex; align-items: center; gap: 3px; padding: 1px 4px 1px 3px; border-radius: 9px;
    background: var(--bg-surface); border: 1px dashed var(--planned); box-shadow: 0 1px 3px rgb(20 14 8 / 0.3);
  }
  .plan b { font: 700 10px/14px var(--font-sans); color: var(--planned-text); min-width: 9px; text-align: center; }
  .dealt {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%, -30%);
    font: 800 18px/1 var(--font-sans); color: #fff;
    padding: 2px 6px; border-radius: 5px; background: #9b2f22; box-shadow: 0 2px 6px rgb(20 14 8 / 0.4);
    animation: land calc(var(--motion-move) * 1.6) var(--motion-move-ease);
  }
  @keyframes land { from { transform: translate(-50%, -90%) scale(1.25); opacity: 0; } }
  .pt {
    position: absolute; right: 3px; bottom: 3px; padding: 0 0.35em; border-radius: 4px;
    font-family: var(--font-sans); font-weight: 700; line-height: 1.35; font-variant-numeric: tabular-nums;
    color: #2a2116; background: #f3ead6; border: 1px solid rgb(20 14 8 / 0.5);
  }
  .pt.up { color: #2f5f26; }
  .pt.hurt { color: #7e2a20; }
  .pt small { font-size: 0.75em; margin-left: 2px; }
  .keywords { position: absolute; left: 3px; bottom: 3px; display: flex; gap: 2px; max-width: 62%; overflow: hidden; }
  .keywords i {
    font: 600 9px/13px var(--font-sans); font-style: normal; text-transform: capitalize; white-space: nowrap;
    padding: 0 4px; border-radius: 3px; color: #f8f1e0; background: rgb(12 9 5 / 0.72);
  }
  .keywords i.granted { box-shadow: inset 0 0 0 1px #9ecb8b; }
  @media (prefers-reduced-motion: reduce) {
    .slot, .tcard { transition: none; }
    .dealt { animation: none; }
  }
</style>
