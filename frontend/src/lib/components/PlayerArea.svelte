<script lang="ts">
  import type { CardState, PlayerState } from '$lib/types';

  import Card from './Card.svelte';
  import CardBack from './CardBack.svelte';
  import PermanentRow from './PermanentRow.svelte';

  // A player's region of the sheet: ruled staves with margin rubrics, the
  // way a score names its staves. The opponent reads hand-first (their hand
  // faces you); the hero reads battlefield-first.
  interface Props {
    label: string;
    player: PlayerState;
    deckName?: string | null;
    opponent?: boolean;
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    onPreviewCard?: (
      card: { name: string | null; power: number | null; toughness: number | null } | null,
    ) => void;
  }

  let {
    label,
    player,
    deckName = null,
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
      power: card.types.is_creature ? card.power : null,
      toughness: card.types.is_creature ? card.toughness : null,
    });
  }
</script>

{#snippet rubric(text: string)}
  <div class="type-rubric pt-2 text-right text-ink-2">
    {text}
  </div>
{/snippet}

{#snippet playerBar()}
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

<section class="py-3" aria-label={`${label}${deckName ? ` — ${deckName}` : ''}`}>
  {#if opponent}
    {@render playerBar()}
  {/if}

  <div class="staves">
    {#if opponent}
      <div class="staff">
        {@render rubric(`Hand (${hiddenHandCount})`)}
        <div class="flex flex-wrap items-end gap-2">
          {#if hiddenHandCount === 0}
            <div class="type-caption px-1 py-1 text-ink-3">Empty hand</div>
          {/if}
          {#each Array(hiddenHandCount) as _, index}
            <div role="img" aria-label={`Hidden card ${index + 1}`}>
              <CardBack />
            </div>
          {/each}
        </div>
      </div>
      <div class="staff">
        {@render rubric('Battlefield')}
        <PermanentRow
          permanents={player.battlefield}
          {focusedIds}
          {clickableTargets}
          {onSelectTarget}
          {onHoverTarget}
          {onPreviewCard}
        />
      </div>
    {:else}
      <div class="staff">
        {@render rubric('Battlefield')}
        <PermanentRow
          permanents={player.battlefield}
          {focusedIds}
          {clickableTargets}
          {onSelectTarget}
          {onHoverTarget}
          {onPreviewCard}
        />
      </div>
      <div class="staff">
        {@render rubric(`Hand (${player.hand.length})`)}
        <div class="flex flex-wrap items-end gap-2">
          {#if player.hand.length === 0}
            <div class="type-caption px-1 py-1 text-ink-3">Empty hand</div>
          {/if}
          {#each player.hand as card}
            <Card
              name={card.name}
              power={card.types.is_creature ? card.power : null}
              toughness={card.types.is_creature ? card.toughness : null}
              focused={focusedIds.has(card.id)}
              clickable={clickableTargets?.has(card.id) ?? false}
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

    <div class="staff">
      {@render rubric(`Graveyard (${player.graveyard.length})`)}
      <div class="flex flex-wrap items-end gap-2">
        {#if player.graveyard.length === 0}
          <div class="type-caption px-1 py-1 text-ink-3">Empty graveyard</div>
        {/if}
        {#each player.graveyard as card}
          <Card
            name={card.name}
            power={card.types.is_creature ? card.power : null}
            toughness={card.types.is_creature ? card.toughness : null}
            focused={focusedIds.has(card.id)}
            spent
            onHoverStart={() => preview(card)}
            onHoverEnd={() => onPreviewCard?.(null)}
          />
        {/each}
      </div>
    </div>

    {#if player.exile.length > 0}
      <div class="staff">
        {@render rubric(`Exile (${player.exile.length})`)}
        <div class="flex flex-wrap items-end gap-2">
          {#each player.exile as card}
            <Card
              name={card.name}
              power={card.types.is_creature ? card.power : null}
              toughness={card.types.is_creature ? card.toughness : null}
              focused={focusedIds.has(card.id)}
              spent
              onHoverStart={() => preview(card)}
              onHoverEnd={() => onPreviewCard?.(null)}
            />
          {/each}
        </div>
      </div>
    {/if}
  </div>

  {#if !opponent}
    {@render playerBar()}
  {/if}
</section>

<style>
  .staves .staff {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr);
    column-gap: 18px;
    align-items: end;
    padding: 10px 0 12px;
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
