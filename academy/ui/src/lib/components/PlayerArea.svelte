<script lang="ts">
  import { revealWithinScrollRegions } from '$lib/scroll-regions';
  import { copyLabel } from '$lib/decision-labels';
  import { effectNote } from '$lib/permanent-notes';

  import type {
    CardPreview,
    CardState,
    CastPermission,
    PermanentState,
    PlayerState,
    PlayerTurnHistory,
    RestrictedMana
  } from '$lib/types';
  import Card from './Card.svelte';
  import CardBack from './CardBack.svelte';
  import Battlefield from './Battlefield.svelte';

  // Player identity and the hero hand remain outside the scrolling zones.
  // Secondary zones expose only cards already permitted by the viewer projection.
  interface Props {
    label: string;
    player: PlayerState;
    /** Every visible permanent at the table, for attachments across seats. */
    tablePermanents: PermanentState[];
    deckName?: string | null;
    definitionNames?: Record<string, string>;
    opponent?: boolean;
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    onPreviewCard?: (card: CardPreview | null) => void;
  }

  let {
    label,
    player,
    tablePermanents,
    deckName = null,
    definitionNames = {},
    opponent = false,
    focusedIds = new Set<number>(),
    clickableTargets = undefined,
    onSelectTarget = undefined,
    onHoverTarget = undefined,
    onPreviewCard = undefined,
  }: Props = $props();

  const hiddenHandCount = $derived(
    player.hand_hidden_count ?? player.zone_counts.HAND ?? player.hand.length,
  );
  function total(counts: Record<string, number> | undefined): number {
    return Object.values(counts ?? {}).reduce((sum, count) => sum + count, 0);
  }
  // Another player's face-down exiled cards are counted, never listed.
  const exileCount = $derived(Math.max(player.zone_counts.EXILE ?? 0, player.exile.length));
  const hiddenExileCount = $derived(exileCount - player.exile.length);

  function exileDescription(card: CardState): string {
    const parts = [
      card.face_down ? 'foretold, face down' : '',
      card.cast_permission ? permissionText(card.cast_permission) : '',
      copyLabel(player.exile, card.id),
    ];
    return parts.filter(Boolean).join(', ');
  }

  // Every term of the permission is shown; whether it can be used right now
  // is the offer list's job.
  function permissionText(permission: CastPermission): string {
    const who = permission.player_id === player.id ? 'its owner' : 'the other player';
    const terms = [permission.cost ? `may be cast by ${who} for ${permission.cost}` : `may be played by ${who}`];
    if (permission.this_turn) terms.push('this turn');
    if (permission.through_next_turn) terms.push('until the end of their next turn');
    if (permission.own_turn_only) terms.push('during their turn');
    if (permission.after_turn !== undefined) terms.push(`after turn ${permission.after_turn}`);
    if (permission.source_permanent_id !== undefined) terms.push('while they control the card that exiled it');
    if (permission.flash) terms.push('as though it had flash');
    if (permission.any_mana) terms.push('with mana of any type');
    if (permission.waterbend) terms.push('waterbending for generic');
    return terms.join(' ');
  }

  function spendingText(restricted: RestrictedMana): string {
    if (restricted.except_nonartifact_spells) return 'anything but nonartifact spells';
    const parts = [];
    if (restricted.spells?.length) parts.push(`${restricted.spells.join(' or ')} spells`);
    if (restricted.abilities?.length) parts.push(`abilities of ${restricted.abilities.join(' or ')} sources`);
    return parts.join(' or ');
  }

  // The bar wears its deck's colors at full Magic saturation — G→W for
  // Allies, U→R for Lessons. Like the banner and the card plates it is a
  // fixed rich world in both modes: vivid ends, a whisper of scrim, and
  // literal ivory text.
  const VIVID: Record<string, string> = {
    W: 'var(--vivid-w)',
    U: 'var(--vivid-u)',
    B: 'var(--vivid-b)',
    R: 'var(--vivid-r)',
    G: 'var(--vivid-g)',
  };
  const pieColors = $derived.by(() => {
    const match = /^([WUBRG]{1,5}) /.exec(deckName ?? '');
    return match ? match[1].split('').map((letter) => VIVID[letter]) : [];
  });
  const barBackground = $derived.by(() => {
    if (pieColors.length === 0) {
      return '';
    }
    const scrim = 'linear-gradient(rgb(15 11 6 / 0.3), rgb(15 11 6 / 0.3))';
    if (pieColors.length === 1) {
      return `${scrim}, ${pieColors[0]}`;
    }
    const last = pieColors.length - 1;
    const stops = pieColors.map(
      (color, index) => `${color} ${Math.round((index / last) * 100)}%`,
    );
    return `${scrim}, linear-gradient(90deg, ${stops.join(', ')})`;
  });

  function preview(card: CardState): void {
    onPreviewCard?.({
      name: card.name,
      text_box: card.text_box,
      power: card.types.is_creature ? card.power : null,
      toughness: card.types.is_creature ? card.toughness : null,
    });
  }
  function manaText(pool: number[]): string {
    return pool.map((amount, index) => amount ? `${amount} {${['W', 'U', 'B', 'R', 'G', 'C'][index]}}` : '').filter(Boolean).join(' · ') || 'Empty';
  }
</script>

{#snippet rubric(text: string)}
  <div class="type-rubric pt-1 text-left text-ink-2">
    {text}
  </div>
{/snippet}

{#snippet playerBar()}
  <details class="player-status type-caption"><summary>{label} status and mana</summary>
  {#if player.mana}
    <p class="type-caption py-1" data-testid="mana-pool">{label} mana: {manaText(player.mana)}
      {#if player.restricted_mana?.length}
        {#each player.restricted_mana as restricted}
          · {manaText(restricted.mana)} only for {spendingText(restricted)}
        {/each}
      {:else if player.lesson_mana?.some(Boolean)} · Restricted mana: {manaText(player.lesson_mana)}{/if}
      <!-- Class rows are subsets of the amounts above, not additional mana. -->
      {#each player.pooled_mana_classes ?? [] as pooled, index}
        {index ? '; ' : ' — including '}{manaText(pooled.mana)}{pooled.restriction ? ` only for ${pooled.restriction}` : ''}{pooled.effect ? ` that puts ${pooled.effect}` : ''}, expiring {pooled.combat ? 'at end of combat' : 'at step end'}
      {/each}
      {#if !player.pooled_mana_classes?.length && player.combat_mana} · {player.combat_mana} mana expires at end of combat{/if}
      {#if player.keeps_red_mana} · Keeps unspent red mana until end of turn{/if}
    </p>
  {:else}<p class="type-caption text-ink-2">Mana details not recorded.</p>{/if}
  {#if player.hand_only_until !== undefined}
    <p class="type-caption py-1" data-testid="hand-only">
      {label} can cast spells only from hand until the other player's next turn.
    </p>
  {/if}
  {#each player.effects ?? [] as effect}
    <p class="type-caption py-1" data-testid="player-effect">{label}: {effectNote(effect)}.</p>
  {/each}
  {#if player.no_maximum_hand_size}
    <p class="type-caption py-1" data-testid="no-maximum-hand-size">
      {label} has no maximum hand size.
    </p>
  {/if}
  {#if player.turn_history || player.spells_cast}
    {@const history: PlayerTurnHistory = player.turn_history ?? {}}
    <p class="type-caption py-1" data-testid="turn-history">
      {label} this turn: {[
        history.attacked && 'attacked',
        history.permanents_sacrificed && `sacrificed ${history.permanents_sacrificed} permanent${history.permanents_sacrificed === 1 ? '' : 's'}`,
        history.land_entered && 'a land entered',
        history.cards_drawn && `drew ${history.cards_drawn} card${history.cards_drawn === 1 ? '' : 's'}`,
        history.bent && `bent ${['water', 'earth', 'fire', 'air'].filter((_, index) => (history.bent ?? 0) & (1 << index)).join(', ')}`,
        history.life_lost && `lost ${history.life_lost} life`,
        player.spells_cast && `cast ${player.spells_cast} spell${player.spells_cast === 1 ? '' : 's'}`,
      ].filter(Boolean).join(' · ')}.
    </p>
  {/if}
  </details>
  <!-- The player bar, in the manner of a chess clock plate: identity at
       one end, life at the other. The opponent's sits above their side of
       the table; yours below yours — the players bracket the battlefield. -->
  <div
    class={`rounded-sm px-4 py-2 ${pieColors.length > 0 ? 'border border-black/30' : 'border border-line bg-field/70'}`}
    style:background={barBackground || undefined}
  >
    <div class="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
      <div class="min-w-0">
        <h2 class={`type-display inline ${pieColors.length > 0 ? 'text-[#f8f1e0] [text-shadow:0_1px_2px_rgb(0_0_0/0.35)]' : 'text-display'}`}>
          {label}{#if deckName}<span class={`type-annotation ${pieColors.length > 0 ? 'text-[#f8f1e0]/80' : 'text-ink-2'}`}> — {deckName}</span>{/if}
        </h2>
      </div>
      <div class={`flex items-baseline gap-4 ${pieColors.length > 0 ? 'text-[#f8f1e0]' : ''}`}>
        <span class={`type-caption ${pieColors.length > 0 ? 'text-[#f8f1e0]/75' : 'text-ink-2'}`}>Library {player.library_count}</span>
        <div
          class={`flex items-baseline gap-1.5 ${focusedIds.has(player.id) ? `rounded outline-2 outline-offset-4 ${pieColors.length > 0 ? 'outline-[#f8f1e0]' : 'outline-action'}` : ''}`}
        >
          <b class="type-numeral [text-shadow:0_1px_2px_rgb(0_0_0/0.3)]">{player.life}</b>
          <span class={`type-rubric ${pieColors.length > 0 ? 'text-[#f8f1e0]/70' : 'text-ink-2'}`}>life</span>
        </div>
      </div>
    </div>
  </div>
{/snippet}

<section class="player-area" aria-label={`${label}${deckName ? ` — ${deckName}` : ''}`}>
  {@render playerBar()}

  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to the scrolling zones) -->
  <div class="staves" role="region" aria-label={`${label} battlefield and zones`} tabindex="0">
    {#if player.main_counts}
      <details class="type-caption py-2" data-testid="decklist-summary">
        <summary>Decklist ({total(player.main_counts)} cards, {total(player.unseen_counts)} unseen)</summary>
        {#each Object.entries(player.main_counts) as [definition, initial]}
          <div>{definitionNames[definition] ?? `Card ${definition}`}: {player.unseen_counts?.[definition] ?? 0} unseen of {initial}</div>
        {/each}
      </details>
    {:else if opponent}
      <div class="type-caption py-2 text-ink-2" data-testid="decklist-summary">Decklist closed</div>
    {/if}
    {#if Object.keys(player.sideboard_counts ?? {}).length > 0}
      <details class="type-caption py-2" data-testid="sideboard-summary">
        <summary>Sideboard ({total(player.remaining_sideboard_counts)} remaining)</summary>
        {#each Object.entries(player.sideboard_counts ?? {}) as [definition, initial]}
          <div>{definitionNames[definition] ?? `Card ${definition}`}: {player.remaining_sideboard_counts?.[definition] ?? 0} of {initial}</div>
        {/each}
      </details>
    {/if}
    {#if Object.keys(player.known_hand ?? {}).length > 0}
      <div class="type-caption py-2" data-testid="known-hand">
        Revealed in hand:
        {#each Object.entries(player.known_hand ?? {}) as [definition, count]}
          <span>{count} × {definitionNames[definition] ?? `Card ${definition}`}</span>
        {/each}
      </div>
    {/if}
    {#if opponent}
      <details class="type-caption text-ink-2">
        <summary>Hand ({hiddenHandCount}) · hidden</summary>
        <div class="hand-cards">
          {#each Array(hiddenHandCount) as _, index}
            <div role="img" aria-label={`Hidden card ${index + 1}`}><CardBack /></div>
          {/each}
        </div>
      </details>
    {/if}
      <div class="staff">
        {@render rubric('Battlefield')}
        <Battlefield
          permanents={player.battlefield}
          {tablePermanents}
          {focusedIds}
          {clickableTargets}
          {onSelectTarget}
          {onHoverTarget}
          {onPreviewCard}
        />
      </div>

    <details open={player.graveyard.some(card => focusedIds.has(card.id))}>
      <summary class="type-caption cursor-pointer">Graveyard ({player.graveyard.length})</summary>
      <div class="flex flex-wrap items-end gap-2">
        {#if player.graveyard.length === 0}
          <div class="type-caption px-1 py-1 text-ink-3">Empty graveyard</div>
        {/if}
        {#each player.graveyard as card}
          <Card
            name={card.name}
              copyDescription={copyLabel(player.graveyard, card.id)}
            power={card.types.is_creature ? card.power : null}
            toughness={card.types.is_creature ? card.toughness : null}
            focused={focusedIds.has(card.id)}
            spent
            clickable={onSelectTarget !== undefined}
            onSelect={() => onSelectTarget?.(card.id)}
            onHoverStart={() => preview(card)}
            onHoverEnd={() => onPreviewCard?.(null)}
          />
        {/each}
      </div>
    </details>

    {#if exileCount > 0}
      <details open={player.exile.some(card => focusedIds.has(card.id))}>
        <summary class="type-caption cursor-pointer">Exile ({exileCount})</summary>
        <div class="flex flex-wrap items-end gap-2">
          {#if hiddenExileCount > 0}
            <div class="type-caption px-1 py-1 text-ink-3">
              {hiddenExileCount} face-down {hiddenExileCount === 1 ? 'card' : 'cards'}
            </div>
          {/if}
          {#each player.exile as card}
            <Card
              name={card.name}
              copyDescription={exileDescription(card)}
              power={card.types.is_creature ? card.power : null}
              toughness={card.types.is_creature ? card.toughness : null}
              focused={focusedIds.has(card.id)}
              spent
              clickable={onSelectTarget !== undefined}
              onSelect={() => onSelectTarget?.(card.id)}
              onHoverStart={() => preview(card)}
              onHoverEnd={() => onPreviewCard?.(null)}
            />
          {/each}
        </div>
      </details>
    {/if}
  </div>

  {#if !opponent}
      <div class="hand-staff">
        {@render rubric(`Hand (${player.hand.length})`)}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to horizontal hand scrolling) -->
        <div role="region" aria-label={`${label} hand`} tabindex="0" class="hand-cards"
          onfocusin={(event) => {
            if (event.target instanceof HTMLElement) revealWithinScrollRegions(event.target);
          }}>
          {#if player.hand.length === 0}
            <div class="type-caption px-1 py-1 text-ink-3">Empty hand</div>
          {/if}
          {#each player.hand as card}
            <Card
              name={card.name}
              copyDescription={copyLabel(player.hand, card.id)}
              power={card.types.is_creature ? card.power : null}
              toughness={card.types.is_creature ? card.toughness : null}
              focused={focusedIds.has(card.id)}
              clickable={onSelectTarget !== undefined}
              onSelect={() => onSelectTarget?.(card.id)}
              onHoverStart={() => {
                onHoverTarget?.(clickableTargets?.has(card.id) ? card.id : null);
                preview(card);
              }}
              onHoverEnd={() => {
                onHoverTarget?.(null);
                onPreviewCard?.(null);
              }}
            />
          {/each}
        </div>
      </div>
  {/if}
</section>

<style>
  .player-area { min-width: 0; padding: .25rem 0; }
  .hand-cards { display: flex; align-items: end; gap: .5rem; overflow-x: auto; overscroll-behavior: contain; padding: .25rem; }
  .hand-cards > :global(*) { flex-shrink: 0; }
  .hand-staff { min-width: 0; }
  @media (min-width: 1024px) and (min-height: 650px) {
    .player-area { display: flex; flex-direction: column; min-height: 0; }
    .player-status { flex-shrink: 0; max-height: 90px; overflow: auto; }
    .staves { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 0 .25rem; }
    .hand-staff { flex-shrink: 0; }
  }
  .staves .staff {
    display: grid;
    grid-template-columns: 1fr;
    column-gap: 18px;
    align-items: end;
    padding: 4px 0;
  }
  .staves .staff + .staff {
    border-top: 1px solid color-mix(in srgb, var(--border) 62%, transparent);
  }
  @media (max-width: 640px) {
    .staves .staff {
      grid-template-columns: 1fr;
      row-gap: 6px;
    }
    .staves .staff > :first-child {
      padding-top: 6px;
      text-align: left;
    }
  }
</style>
