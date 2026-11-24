<script lang="ts">
  /**
   * A player's plate: who they are, life, zones and floating mana. The plate
   * is the player as a target, so offers on the player outline it.
   */
  import { floatingTotal, manaText, pooledManaText, spendPhrase, spendingText } from '$lib/mana';
  import { poolSymbols } from './payment';
  import Pips from './Pips.svelte';
  import type { CardState, PlayerState } from '$lib/types';
  import type { PlacedOffer } from './view';

  interface Props {
    player: PlayerState;
    name: string;
    deck: string | null;
    offers: PlacedOffer[];
    live: boolean;
    /** Damage this player was dealt since your last move. */
    damage?: number;
    /** Set while this player is choosing: "choosing…". */
    choosing?: boolean;
    side: 'you' | 'opponent';
    onactivate: (element: HTMLElement) => void;
    onzone: (zone: string, cards: CardState[], element: HTMLElement) => void;
  }

  let { player, name, deck, offers, live, damage = 0, choosing = false, side, onactivate, onzone }: Props = $props();

  const hand = $derived(player.zone_counts.HAND ?? player.hand.length + (player.hand_hidden_count ?? 0));
  const zones = $derived([
    { key: 'Hand', count: hand, cards: null },
    { key: 'Library', count: player.library_count, cards: null },
    { key: 'Graveyard', count: Math.max(player.zone_counts.GRAVEYARD ?? 0, player.graveyard.length), cards: player.graveyard },
    { key: 'Exile', count: Math.max(player.zone_counts.EXILE ?? 0, player.exile.length), cards: player.exile },
  ]);
  const total = $derived(floatingTotal(player));
  const pool = $derived(total.some(Boolean) ? manaText(total) : null);
  // Mana that is not plain (restricted, with an effect when spent, or lasting
  // only to end of combat) says so beside the pips; each is part of the total.
  const special = $derived([
    ...(player.restricted_mana ?? []).map(part => `${manaText(part.mana)} ${spendPhrase(spendingText(part))}`),
    ...(player.pooled_mana_classes ?? []).map(pooledManaText),
  ]);
  const counters = $derived(Object.entries(player.counters ?? {}).map(([kind, count]) => `${count} ${kind}`).join(', '));
  const label = $derived(`${name}, ${player.life} life${counters ? `, ${counters}` : ''}${offers.length ? ` — ${offers.map(placed => `${placed.offer.label} (${placed.number})`).join('; ')}` : ''}`);
</script>

<section class="plate" aria-label={`${name}'s plate`} data-testid="player-plate" data-player-index={player.player_index} data-plate-side={side}>
  <button
    type="button"
    class="who"
    class:legal={live && offers.length > 0}
    aria-disabled={!offers.length}
    data-offer-count={offers.length}
    aria-label={label}
    data-testid="player-target"
    onclick={event => onactivate(event.currentTarget)}
  >
    <span class="type-title text-display block truncate">{name}</span>
    {#if deck}<span class="type-caption block truncate text-ink-2">{deck}</span>{/if}
    {#if choosing}<span class="type-label choosing" data-testid="plate-choosing">Choosing…</span>{/if}
    <span class="life"><span class="numeral">{player.life}</span><span class="type-caption text-ink-2">life</span>
      {#if damage}{#key damage}<span class="dealt" data-testid="player-damage">−{damage}<span class="sr-only"> since your last move</span></span>{/key}{/if}</span>
    {#if counters}<span class="type-caption">{counters}</span>{/if}
  </button>
  <ul class="zones">
    {#each zones as zone (zone.key)}
      <li>
        {#if zone.cards && zone.cards.length}
          <button type="button" class="zone-open" onclick={event => onzone(zone.key, zone.cards!, event.currentTarget)}
            aria-label={`${zone.key}: ${zone.count}. Show cards`}>
            <span>{zone.key}</span><b>{zone.count}</b>
          </button>
        {:else}
          <span>{zone.key}</span><b>{zone.count}</b>
        {/if}
      </li>
    {/each}
  </ul>
  <p class="type-caption pool" class:empty={!pool} data-testid="mana-pool">
    {#if pool}<span class="text-ink-3" aria-hidden="true">Floating</span><span aria-hidden="true"><Pips cost={poolSymbols(total)} size={16} /></span><span class="sr-only">Floating: {pool}</span>{:else}No floating mana{/if}
  </p>
  {#each special as text, index (index)}<p class="type-caption pool-class" data-testid="mana-pool-class">{text}</p>{/each}
</section>

<style>
  .plate { display: flex; flex-direction: column; gap: 6px; min-width: 0; min-height: 0; overflow-y: auto; padding: 10px 10px 8px; }
  /* A short half scrolls the plate; its parts keep their height rather than overlap. */
  .plate > * { flex-shrink: 0; }
  .who { display: block; text-align: left; border-radius: var(--radius-sm); padding: 2px 4px; margin: -2px -4px; background: none; border: 0; color: inherit; }
  .who[aria-disabled=true] { cursor: default; }
  .who.legal { box-shadow: 0 0 0 2px var(--legal); cursor: pointer; }
  .choosing { display: block; margin-top: 2px; color: var(--warning-text); }
  .life { display: flex; align-items: baseline; gap: 6px; margin-top: 2px; }
  .numeral { font-family: var(--font-serif); font-size: 40px; font-weight: 600; line-height: 44px; color: var(--text); font-variant-numeric: lining-nums tabular-nums; }
  .zones { display: grid; grid-template-columns: 1fr; gap: 0; margin: 0; padding-top: 4px; border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent); }
  .zones > li, .zone-open {
    display: flex; justify-content: space-between; gap: 4px; padding: 0 4px; border-radius: var(--radius-sm);
    font-family: var(--font-sans); font-size: 12px; line-height: 20px; color: var(--text-secondary); min-width: 0;
  }
  .zones > li:has(.zone-open) { padding: 0; }
  .zone-open { width: 100%; height: 20px; min-height: 0; cursor: pointer; text-decoration: underline dotted color-mix(in srgb, currentColor 40%, transparent); text-underline-offset: 3px; }
  .zone-open:hover, .zone-open:focus-visible { background: var(--bg-field); color: var(--text); }
  .zones span { overflow: hidden; text-overflow: ellipsis; }
  .zones b { font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
  .pool { color: var(--text-secondary); display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }
  .dealt {
    align-self: center; margin-left: auto; padding: 0 5px; border-radius: var(--radius-sm);
    font: 600 16px/24px var(--font-sans); color: var(--bg); background: var(--damage);
    animation: land calc(var(--motion-move) * 1.6) var(--motion-move-ease);
  }
  @keyframes land { from { transform: translateY(-40%) scale(1.2); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .dealt { animation: none; } }
  .pool.empty { color: var(--text-secondary); }
  .pool-class { color: var(--text-secondary); margin-top: -3px; }
</style>
