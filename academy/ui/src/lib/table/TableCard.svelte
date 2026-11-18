<script lang="ts">
  /**
   * One card on the table: a permanent, a card in hand, a spell on the stack
   * or a tray candidate. It is a button whenever the viewer may inspect it;
   * its outline says a legal offer acts on it, and its accessible name says
   * which. The art is a crop: rules text lives in the preview.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import type { CardState, PermanentState } from '$lib/types';
  import { CARD_RATIO, type PlacedOffer } from './view';

  interface Props {
    name: string;
    width: number;
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
    /** Copies in this card's pile, shown on the top copy. */
    count?: number | null;
    onactivate?: (element: HTMLElement) => void;
    onpreview?: (element: HTMLElement | null) => void;
  }

  let {
    name, width, permanent = null, card = null, offers = [], live = false, asked = false,
    toward = null, blocks = null, side = 'you', touched = false, count = null, onactivate, onpreview,
  }: Props = $props();

  const height = $derived(Math.round(width / CARD_RATIO));
  const tapped = $derived(permanent?.tapped ?? false);
  const creature = $derived(permanent ? permanent.types?.is_creature ?? permanent.power !== null : card?.types.is_creature ?? false);
  const power = $derived(permanent?.power ?? card?.power ?? null);
  const toughness = $derived(permanent?.toughness ?? card?.toughness ?? null);
  const counters = $derived(Object.entries({
    ...(permanent?.plus1_counters ? { '+1/+1': permanent.plus1_counters } : {}),
    ...(permanent?.counters ?? {}),
  }).filter(([, count]) => count > 0));
  const compact = $derived(width < 64);

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
    ...counters.map(([kind, count]) => `${count} ${kind} counter${count === 1 ? '' : 's'}`),
    permanent?.is_token ? 'token' : '',
  ].filter(Boolean).join(', '));
  const label = $derived([
    `${name}${creature && power !== null ? ` ${power}/${toughness}` : ''}`,
    facts,
    offers.length === 1 ? `${offers[0].offer.label} (${offers[0].number})`
      : offers.length > 1 ? `${offers.length} choices: ${offers.map(placed => placed.offer.label).join('; ')}` : '',
  ].filter(Boolean).join(' — '));
</script>

<!-- The slot reserves a tapped card's turned footprint so rows never reflow on tap. -->
<div
  class="slot"
  class:tapped
  class:lean-up={permanent?.attacking && toward === 'up'}
  class:lean-down={permanent?.attacking && toward === 'down'}
  style:width={`${tapped ? height : width}px`}
  style:height={`${height}px`}
>
  <button
    type="button"
    class="tcard"
    class:legal={live && offers.length > 0}
    class:asked
    class:attacking={permanent?.attacking}
    class:has-art={art !== null}
    class:touched
    data-testid="table-card"
    data-travel-id={permanent ? `perm:${permanent.id}` : card ? `card:${card.id}` : undefined}
    data-travel-name={name}
    data-travel-side={side}
    data-object-id={permanent?.id ?? card?.id}
    data-offer-count={offers.length}
    data-card-name={name}
    aria-label={label}
    style:width={`${width}px`}
    style:height={`${height}px`}
    style:background-image={art ? `url('${art}')` : undefined}
    onclick={event => onactivate?.(event.currentTarget)}
    onmouseenter={event => onpreview?.(event.currentTarget)}
    onmouseleave={() => onpreview?.(null)}
    onfocus={event => onpreview?.(event.currentTarget)}
    onblur={() => onpreview?.(null)}
  >
    {#if !art}<span class="face-name" aria-hidden="true">{name}</span>{/if}
    {#if card && !permanent && card.mana_value > 0 && !card.types.is_land}
      <span class="mv" aria-hidden="true">{card.mana_value}</span>
    {/if}
    {#if counters.length && !compact}
      <span class="counters" aria-hidden="true">{counters.map(([kind, count]) => `${kind === '+1/+1' ? '+1' : kind} ×${count}`).join(' · ')}</span>
    {/if}
    {#if art && !compact}<span class="name" aria-hidden="true">{name}</span>{/if}
    {#if creature && power !== null}
      <span class="pt" class:hurt={(permanent?.damage ?? 0) > 0} aria-hidden="true">{power}/{toughness}{#if permanent?.damage}<small>−{permanent.damage}</small>{/if}</span>
    {/if}
    {#if offers.length > 1 && live}<span class="many" aria-hidden="true">{offers.length}</span>{/if}
  </button>
  {#if count}<span class="count" aria-hidden="true" style:top={`${tapped ? Math.round((height - width) / 2) + 3 : 3}px`}>×{count}</span>{/if}
</div>

<style>
  .slot { position: relative; flex: none; display: grid; place-items: center; transition: transform var(--motion-move) var(--motion-move-ease); }
  /* An attacker steps toward the player it attacks: a fifth of its height, less when lines are stacked. */
  .slot.lean-up { transform: translateY(calc(-1 * var(--lean, 20%))); }
  .slot.lean-down { transform: translateY(var(--lean, 20%)); }
  .tcard {
    position: relative; display: block; overflow: hidden; padding: 0; cursor: pointer;
    border-radius: max(4px, 6%); border: 1px solid rgb(20 14 8 / 0.55);
    background: var(--bg-muted) center 30% / cover no-repeat;
    box-shadow: 0 1px 2px rgb(20 14 8 / 0.25);
    transition: transform var(--motion-move) var(--motion-move-ease), box-shadow var(--motion-fast) ease-out, filter var(--motion-fast);
  }
  .tapped .tcard { transform: rotate(90deg); }
  .tapped .tcard.has-art { filter: saturate(0.55) brightness(0.92); }
  .tcard:focus-visible { outline: 2px solid var(--text); outline-offset: 3px; }
  .tcard.attacking { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 6px 14px rgb(20 14 8 / 0.3); }
  .tcard.touched { box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 55%, transparent), 0 1px 2px rgb(20 14 8 / 0.25); }
  .tcard.legal { box-shadow: 0 0 0 2px var(--legal), 0 0 12px color-mix(in srgb, var(--legal) 45%, transparent); }
  .tcard.legal:hover { box-shadow: 0 0 0 3px var(--legal), 0 0 16px color-mix(in srgb, var(--legal) 60%, transparent); }
  .tcard.asked { box-shadow: 0 0 0 2px var(--accent), 0 8px 18px rgb(20 14 8 / 0.35); transform: translateY(-8px); }
  .face-name {
    position: absolute; inset: 6px; font-family: var(--font-serif); font-weight: 600; font-size: 12px; line-height: 14px;
    color: var(--text); text-align: left; overflow: hidden;
  }
  .name {
    position: absolute; left: 0; right: 0; bottom: 0; padding: 10px 5px 3px; text-align: left;
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 600; line-height: 13px; color: #f8f1e0;
    background: linear-gradient(transparent, rgb(12 9 5 / 0.82) 45%);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .pt {
    position: absolute; right: 3px; bottom: 3px; padding: 0 4px; border-radius: 4px;
    font-family: var(--font-sans); font-size: 11px; font-weight: 700; line-height: 16px;
    color: #2a2116; background: #f3ead6; border: 1px solid rgb(20 14 8 / 0.5);
  }
  .pt.hurt { color: #7e2a20; }
  .pt small { font-size: 9px; margin-left: 2px; }
  .mv, .many {
    position: absolute; top: 3px; right: 3px; min-width: 17px; height: 17px; border-radius: 9px; padding: 0 4px;
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 700; line-height: 15px; text-align: center;
    color: #2a2116; background: #e6dcc4; border: 1px solid rgb(20 14 8 / 0.5);
  }
  .many { right: auto; left: 3px; color: #f8f1e0; background: var(--legal); }
  .count {
    position: absolute; right: 3px; z-index: 1; padding: 0 5px; border-radius: 9px;
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 700; line-height: 16px;
    color: #f8f1e0; background: rgb(12 9 5 / 0.78);
  }
  .counters {
    position: absolute; top: 3px; left: 3px; right: 3px; padding: 0 4px; border-radius: 4px; text-align: left;
    font-family: var(--font-sans); font-size: 9.5px; font-weight: 700; line-height: 14px;
    color: #f8f1e0; background: rgb(12 9 5 / 0.72); white-space: nowrap; overflow: hidden;
  }
  @media (prefers-reduced-motion: reduce) {
    .slot, .tcard { transition: none; }
  }
</style>
