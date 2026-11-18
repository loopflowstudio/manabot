<script lang="ts">
  /**
   * A player's plate: who they are, life, zones and floating mana. The plate
   * is the player as a target, so offers on the player outline it.
   */
  import { floatingTotal, manaText, pooledManaText, spendingText } from '$lib/mana';
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
    active: boolean;
    /** Damage this player was dealt since your last move. */
    damage?: number;
    /** Set while this player is choosing: "choosing…". */
    choosing?: boolean;
    side: 'you' | 'opponent';
    onactivate: (element: HTMLElement) => void;
    onzone: (zone: string, cards: CardState[], element: HTMLElement) => void;
  }

  let { player, name, deck, offers, live, active, damage = 0, choosing = false, side, onactivate, onzone }: Props = $props();

  const hand = $derived(player.hand_hidden_count ?? player.zone_counts.HAND ?? player.hand.length);
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
    ...(player.restricted_mana ?? []).map(part => `${manaText(part.mana)} only for ${spendingText(part)}`),
    ...(player.pooled_mana_classes ?? []).map(pooledManaText),
  ]);
  const label = $derived(`${name}, ${player.life} life${offers.length ? ` — ${offers.map(placed => `${placed.offer.label} (${placed.number})`).join('; ')}` : ''}`);
</script>

<section class="plate" class:active aria-label={`${name}'s plate`} data-testid="player-plate" data-player-index={player.player_index} data-plate-side={side}>
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
    {#if choosing}<span class="type-rubric choosing" data-testid="plate-choosing"><i aria-hidden="true"></i>choosing…</span>{/if}
    <span class="life"><span class="numeral">{player.life}</span><span class="type-rubric text-ink-3">life</span>
      {#if damage}{#key damage}<span class="dealt" data-testid="player-damage">−{damage}<span class="sr-only"> since your last move</span></span>{/key}{/if}</span>
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
    {#if pool}<span class="text-ink-3" aria-hidden="true">Floating</span><span aria-hidden="true"><Pips cost={poolSymbols(total)} size={15} /></span><span class="sr-only">Floating: {pool}</span>{:else}No floating mana{/if}
  </p>
  {#each special as text, index (index)}<p class="type-caption pool-class" data-testid="mana-pool-class">{text}</p>{/each}
</section>

<style>
  .plate { display: flex; flex-direction: column; gap: 6px; min-width: 0; padding: 10px 10px 8px; }
  .plate.active { box-shadow: inset 3px 0 0 var(--warning); }
  .who { display: block; text-align: left; border-radius: 6px; padding: 2px 4px; margin: -2px -4px; background: none; border: 0; color: inherit; }
  .who[aria-disabled=true] { cursor: default; }
  .who.legal { box-shadow: 0 0 0 2px var(--legal); cursor: pointer; }
  .choosing { display: flex; align-items: center; gap: 5px; margin-top: 2px; color: var(--warning-text, var(--text-secondary)); }
  .choosing i { width: 6px; height: 6px; border-radius: 3px; background: var(--warning); animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: 0.25; } }
  @media (prefers-reduced-motion: reduce) { .choosing i { animation: none; } }
  .life { display: flex; align-items: baseline; gap: 6px; margin-top: 2px; }
  .numeral { font-family: var(--font-serif); font-size: 34px; font-weight: 600; line-height: 36px; color: var(--text); font-variant-numeric: tabular-nums; }
  .zones { display: grid; grid-template-columns: 1fr; gap: 3px; margin: 0; }
  .zones > li, .zone-open {
    display: flex; justify-content: space-between; gap: 4px; padding: 1px 6px; border: 1px solid var(--border); border-radius: 4px;
    font-family: var(--font-sans); font-size: 11.5px; line-height: 18px; color: var(--text-secondary); background: var(--bg-field); min-width: 0;
  }
  .zones > li:has(.zone-open) { padding: 0; border: 0; background: none; }
  .zone-open { width: 100%; cursor: pointer; }
  .zone-open:hover { border-color: var(--accent); }
  .zones span { overflow: hidden; text-overflow: ellipsis; }
  .zones b { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
  .pool { color: var(--text-secondary); display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }
  .dealt {
    align-self: center; margin-left: auto; padding: 0 5px; border-radius: 4px;
    font: 800 15px/20px var(--font-sans); color: #fff; background: #9b2f22;
    animation: land calc(var(--motion-move) * 1.6) var(--motion-move-ease);
  }
  @keyframes land { from { transform: translateY(-40%) scale(1.2); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .dealt { animation: none; } }
  .pool.empty { color: var(--text-secondary); }
  .pool-class { color: var(--text-secondary); font-size: 11px; line-height: 14px; margin-top: -3px; }
</style>
