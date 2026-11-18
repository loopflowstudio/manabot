<script lang="ts">
  /**
   * A player's plate: who they are, life, zones and floating mana. The plate
   * is the player as a target, so offers on the player outline it.
   */
  import { manaText } from '$lib/mana';
  import type { CardState, PlayerState } from '$lib/types';
  import type { PlacedOffer } from './view';

  interface Props {
    player: PlayerState;
    name: string;
    deck: string | null;
    offers: PlacedOffer[];
    live: boolean;
    active: boolean;
    onactivate: (element: HTMLElement) => void;
    onzone: (zone: string, cards: CardState[], element: HTMLElement) => void;
  }

  let { player, name, deck, offers, live, active, onactivate, onzone }: Props = $props();

  const hand = $derived(player.hand_hidden_count ?? player.zone_counts.HAND ?? player.hand.length);
  const zones = $derived([
    { key: 'Hand', count: hand, cards: null },
    { key: 'Library', count: player.library_count, cards: null },
    { key: 'Graveyard', count: Math.max(player.zone_counts.GRAVEYARD ?? 0, player.graveyard.length), cards: player.graveyard },
    { key: 'Exile', count: Math.max(player.zone_counts.EXILE ?? 0, player.exile.length), cards: player.exile },
  ]);
  const pool = $derived(player.mana?.some(Boolean) ? manaText(player.mana) : null);
  const label = $derived(`${name}, ${player.life} life${offers.length ? ` — ${offers.map(placed => `${placed.offer.label} (${placed.number})`).join('; ')}` : ''}`);
</script>

<section class="plate" class:active aria-label={`${name}'s plate`} data-testid="player-plate" data-player-index={player.player_index}>
  <button
    type="button"
    class="who"
    class:legal={live && offers.length > 0}
    disabled={!offers.length}
    aria-label={label}
    data-testid="player-target"
    onclick={event => onactivate(event.currentTarget)}
  >
    <span class="type-title text-display block truncate">{name}</span>
    {#if deck}<span class="type-caption block truncate text-ink-2">{deck}</span>{/if}
    <span class="life"><span class="numeral">{player.life}</span><span class="type-rubric text-ink-3">life</span></span>
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
  <p class="type-caption pool" class:empty={!pool} data-testid="mana-pool">{pool ? `Floating: ${pool}` : 'No floating mana'}</p>
</section>

<style>
  .plate { display: flex; flex-direction: column; gap: 6px; min-width: 0; padding: 10px 10px 8px; }
  .plate.active { box-shadow: inset 3px 0 0 var(--warning); }
  .who { display: block; text-align: left; border-radius: 6px; padding: 2px 4px; margin: -2px -4px; background: none; border: 0; color: inherit; }
  .who:disabled { cursor: default; }
  .who.legal { box-shadow: 0 0 0 2px var(--legal); cursor: pointer; }
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
  .pool { color: var(--text-secondary); }
  .pool.empty { color: var(--text-secondary); }
</style>
