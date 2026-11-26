<script lang="ts">
  /**
   * A player's plate: who they are, life, zones and floating mana. The plate
   * is the player as a target, so offers on the player outline it.
   */
  import { floatingTotal, manaText, pooledManaTerms, spendPhrase, spendingText } from '$lib/mana';
  import { poolSymbols } from './payment';
  import Pips from './Pips.svelte';
  import type { CardState, PlayerState } from '$lib/types';
  import type { PlacedOffer } from './view';

  interface Props {
    player: PlayerState;
    name: string;
    definitionNames?: Record<string, string>;
    caption: string | null;
    offers: PlacedOffer[];
    live: boolean;
    /** Damage this player was dealt since your last move. */
    damage?: number;
    /** Signed net life change for this player at the active historical/walk stop (`< 0` loss/damage, `> 0` gain). */
    stopLifeDelta?: number;
    /** Set while this player is choosing: "choosing…". */
    choosing?: boolean;
    /** Highlighted while hovering a target choice naming this player. */
    pointed?: boolean;
    side: 'you' | 'opponent';
    drop?: boolean;
    onactivate: (element: HTMLElement) => void;
    onzone: (zone: string, cards: CardState[], element: HTMLElement) => void;
    ondeck?: () => void;
    deckOpen?: boolean;
  }

  let { drop = false, player, name, definitionNames, caption, offers, live, damage = 0, stopLifeDelta = 0, choosing = false, pointed = false, side, onactivate, onzone, ondeck, deckOpen = false }: Props = $props();

  const hand = $derived(player.zone_counts.HAND ?? player.hand.length + (player.hand_hidden_count ?? 0));
  // An opponent's hand cards this viewer may see (Show both hands, or a card
  // revealed to it) open like a graveyard; your own are the hand strip.
  const shownHand = $derived(side === 'opponent' && player.hand.length ? player.hand : null);
  const zones = $derived<{ key: string; count: number; cards: CardState[] | null; shown?: number }[]>([
    { key: 'Hand', count: hand, cards: shownHand, shown: shownHand?.length },
    { key: 'Library', count: player.library_count, cards: null },
    { key: 'Graveyard', count: Math.max(player.zone_counts.GRAVEYARD ?? 0, player.graveyard.length), cards: player.graveyard },
    { key: 'Exile', count: Math.max(player.zone_counts.EXILE ?? 0, player.exile.length), cards: player.exile },
  ]);
  const knownHand = $derived(Object.entries(player.known_hand ?? {}).flatMap(([id, count]) =>
    definitionNames?.[id] && count > 0 ? [`${count} × ${definitionNames[id]}`] : []));
  const total = $derived(floatingTotal(player));
  const pool = $derived(total.some(Boolean) ? manaText(total) : null);
  // Mana that is not plain (restricted, with an effect when spent, or lasting
  // only to end of combat) says so beside the pips; each is part of the total.
  const special = $derived([
    ...(player.restricted_mana ?? []).map(part => ({ mana: part.mana, terms: spendPhrase(spendingText(part)) })),
    ...(player.pooled_mana_classes ?? []).map(pool => ({ mana: pool.mana, terms: pooledManaTerms(pool) })),
  ]);
  /** Floating mana no class line describes: plain mana, by colour. */
  const plain = $derived(total.map((amount, color) => Math.max(0, amount - special.reduce((sum, part) => sum + (part.mana[color] ?? 0), 0))));
  const hasPlain = $derived(plain.some(Boolean));
  const counters = $derived(Object.entries(player.counters ?? {}).map(([kind, count]) => `${count} ${kind}`).join(', '));
  const label = $derived(`${name}, ${player.life} life${player.hexproof ? ', hexproof' : ''}${counters ? `, ${counters}` : ''}${offers.length ? ` — ${offers.map(placed => `${placed.offer.label} (${placed.number})`).join('; ')}` : ''}`);
</script>

<section class="plate" aria-label={`${name}'s plate`} data-testid="player-plate" data-player-index={player.player_index} data-plate-side={side}>
  <button
    type="button"
    class="who"
    class:legal={live && offers.length > 0}
    class:pointed
    aria-disabled={!offers.length}
    data-offer-count={offers.length}
    aria-label={label}
    data-drop={drop || undefined}
    data-testid="player-target"
    onclick={event => onactivate(event.currentTarget)}
  >
    <span class="type-title text-display block truncate">{name}</span>
    {#if caption}<span class="type-caption caption text-ink-2">{caption}</span>{/if}
    {#if choosing}<span class="type-label choosing" data-testid="plate-choosing">Choosing…</span>{/if}
    <span class="life"><span class="numeral">{player.life}</span><span class="type-caption text-ink-2">life</span>
      {#if damage}{#key damage}<span class="dealt" data-testid="player-damage">−{damage}<span class="sr-only"> since your last move</span></span>{/key}{:else if stopLifeDelta !== 0}{#key stopLifeDelta}<span class="dealt" class:gain={stopLifeDelta > 0} data-testid="stop-delta" data-delta-kind={stopLifeDelta > 0 ? 'gain' : 'damage'}>{stopLifeDelta > 0 ? `+${stopLifeDelta}` : `−${Math.abs(stopLifeDelta)}`}</span>{/key}{/if}</span>
    {#if player.hexproof}<span class="type-caption">Hexproof</span>{/if}
    {#if counters}<span class="type-caption">{counters}</span>{/if}
  </button>
  <ul class="zones">
    {#each zones as zone (zone.key)}
      <li data-graveyard-side={zone.key === 'Graveyard' ? side : undefined}>
        {#if zone.cards && zone.cards.length}
          <button type="button" class="zone-open" onclick={event => onzone(zone.key, zone.cards!, event.currentTarget)}
            aria-label={`${zone.key}: ${zone.count}${zone.shown !== undefined && zone.shown < zone.count ? `, ${zone.shown} shown` : ''}. Show cards`}
            data-testid={zone.key === 'Hand' ? 'shown-hand' : undefined}>
            <span>{zone.key}</span><b>{zone.shown !== undefined && zone.shown < zone.count ? `${zone.shown} of ${zone.count}` : zone.count}</b>
          </button>
        {:else if zone.key === 'Library' && ondeck}
          <button type="button" class="zone-open" aria-pressed={deckOpen} onclick={ondeck}
            data-testid={`plate-deck-toggle-${side}`}
            aria-label={`${zone.key}: ${zone.count}. Toggle deck tracker`}>
            <span>{zone.key}</span><b>{zone.count}</b>
          </button>
        {:else}
          <span>{zone.key}</span><b>{zone.count}</b>
        {/if}
      </li>
    {/each}
  </ul>
  {#if knownHand.length}
    <details class="type-caption"><summary>Known hand cards</summary><ul>{#each knownHand as card}<li>{card}</li>{/each}</ul></details>
  {/if}
  <p class="type-caption pool" class:empty={!pool} data-testid="mana-pool">
    {#if pool}
      <span class="text-ink-3" aria-hidden="true">Floating</span><span class="sr-only">Floating: {pool}</span>
      {#if hasPlain}<span aria-hidden="true"> <Pips cost={poolSymbols(plain)} size={16} /></span>{/if}
      <!-- Each class of mana is its pips and what sets it apart; with no plain mana the first joins the Floating line. -->
      {#each special as part, index (index)}
        <span class="pool-class" class:own-line={index > 0 || hasPlain} data-testid="mana-pool-class">{' '}<span aria-hidden="true"><Pips cost={poolSymbols(part.mana)} size={14} /></span><span class="sr-only">{manaText(part.mana)}{' '}</span>{part.terms}</span>
      {/each}
    {:else}No floating mana{/if}
  </p>
</section>

<style>
  .plate { display: flex; flex-direction: column; gap: 4px; min-width: 0; min-height: 0; overflow-y: auto; padding: 10px 10px 8px; }
  /* A short half scrolls the plate; its parts keep their height rather than overlap. */
  .plate > * { flex-shrink: 0; }
  .who { display: block; text-align: left; border-radius: var(--radius-sm); padding: 2px 4px; margin: -2px -4px; background: none; border: 0; color: inherit; }
  .who[aria-disabled=true] { cursor: default; }
  .who.legal { box-shadow: 0 0 0 2px var(--legal); cursor: pointer; }
  .who.pointed { outline: 2px dashed var(--damage); outline-offset: 2px; box-shadow: 0 0 0 2px var(--accent); }
  /* Kind and deck wrap to a second line rather than cut a long deck name. */
  .caption { overflow-wrap: anywhere; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; }
  .choosing { display: block; margin-top: 2px; color: var(--warning-text); }
  .life { display: flex; align-items: baseline; gap: 6px; margin-top: 2px; }
  .numeral { font-family: var(--font-serif); font-size: 40px; font-weight: 600; line-height: 44px; color: var(--text); font-variant-numeric: lining-nums tabular-nums; }
  .zones { display: grid; grid-template-columns: 1fr; gap: 0; margin: 0; padding-top: 4px; border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent); }
  .zones > li, .zone-open {
    display: flex; justify-content: space-between; gap: 4px; padding: 1px 4px; border-radius: var(--radius-sm);
    font-family: var(--font-sans); font-size: 12px; line-height: 16px; color: var(--text-secondary); min-width: 0;
  }
  .zones > li:has(.zone-open) { padding: 0; }
  .zone-open { width: 100%; height: 18px; min-height: 0; cursor: pointer; text-decoration: underline dotted color-mix(in srgb, currentColor 40%, transparent); text-underline-offset: 3px; }
  .zone-open:hover, .zone-open:focus-visible { background: var(--bg-field); color: var(--text); }
  .zones span { overflow: hidden; text-overflow: ellipsis; }
  .zones b { font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
  /* Floating mana reads as one wrapped sentence: the label, plain pips, then each class of mana. */
  .pool { color: var(--text-secondary); overflow-wrap: anywhere; }
  .pool :global(.pips) { display: inline-flex; vertical-align: -3px; margin-right: 2px; }
  .dealt {
    align-self: center; margin-left: auto; padding: 0 5px; border-radius: var(--radius-sm);
    font: 600 16px/24px var(--font-sans); color: var(--bg); background: var(--damage);
    animation: land var(--motion-move) var(--motion-move-ease);
  }
  .dealt.gain { background: var(--gain); }
  @keyframes land { from { transform: translateY(-40%) scale(1.2); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .dealt { animation: none; } }
  .pool.empty { color: var(--text-secondary); }
  .pool-class.own-line { display: block; }
</style>
