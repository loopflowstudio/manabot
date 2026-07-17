<script lang="ts">
  import CardImage from './CardImage.svelte';

  // A printed plate and the things a print shop does to it: turned and
  // re-set (tapped), still wet (summoning sick), tallied (damage), marked
  // for the printer (targeted), reproduced as an etching (spent), counted
  // in brass (+1/+1).
  interface Props {
    name: string;
    power?: number | null;
    toughness?: number | null;
    focused?: boolean;
    clickable?: boolean;
    tapped?: boolean;
    dimmed?: boolean;
    damage?: number;
    counters?: number;
    spent?: boolean;
    size?: 'small' | 'normal';
    onSelect?: () => void;
    onHoverStart?: () => void;
    onHoverEnd?: () => void;
  }

  let {
    name,
    power = null,
    toughness = null,
    focused = false,
    clickable = false,
    tapped = false,
    dimmed = false,
    damage = 0,
    counters = 0,
    spent = false,
    size = 'small',
    onSelect = undefined,
    onHoverStart = undefined,
    onHoverEnd = undefined,
  }: Props = $props();

  // Tapped cards are landscape: the footprint turns with the card, so
  // rows never shingle and names never read sideways.
  const dims = $derived(
    size === 'normal'
      ? tapped
        ? 'w-56 aspect-[7/5]'
        : 'w-40 aspect-[5/7]'
      : tapped
        ? 'w-28 aspect-[7/5]'
        : 'w-20 aspect-[5/7]',
  );

  // Visual states are also stated in the accessible name.
  const stateNotes = $derived(
    [
      tapped ? 'tapped' : null,
      dimmed ? 'summoning sick' : null,
      damage > 0 ? `${damage} damage` : null,
      counters > 0 ? `${counters} +1/+1 counter${counters > 1 ? 's' : ''}` : null,
    ].filter((note): note is string => note !== null),
  );
  const accessibleName = $derived(
    stateNotes.length > 0 ? `${name} (${stateNotes.join(', ')})` : name,
  );

  const tallies = $derived(Math.min(damage, 5));
</script>

<button
  type="button"
  aria-label={accessibleName}
  aria-disabled={!clickable}
  tabindex={clickable ? 0 : -1}
  class={`group relative ${dims} flex-none overflow-visible rounded-lg text-left transition focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-action ${clickable ? 'cursor-pointer hover:-translate-y-1' : 'cursor-default'} ${spent ? 'spent' : ''}`}
  onclick={() => {
    if (clickable) {
      onSelect?.();
    }
  }}
  onmouseenter={() => onHoverStart?.()}
  onmouseleave={() => onHoverEnd?.()}
  onfocus={() => onHoverStart?.()}
  onblur={() => onHoverEnd?.()}
>
  <div
    class={`absolute inset-0 overflow-hidden rounded-lg border border-black/50 shadow ${tapped ? 'shadow-none' : ''} ${clickable ? 'group-hover:border-action' : ''}`}
  >
    <CardImage
      {name}
      {size}
      landscape={tapped}
      sick={dimmed}
      {spent}
      alt={name}
      className="h-full w-full"
    >
      <div class="flex h-full w-full flex-col justify-between bg-[#10131a] p-2 text-left text-[11px] text-[#faf8f5]">
        <div class="font-semibold leading-tight">{name}</div>
      </div>
    </CardImage>
    {#if tapped}
      <!-- The card rests: one step less light once it has spent itself. -->
      <div class="pointer-events-none absolute inset-0 bg-[rgb(10_13_20/0.18)]" aria-hidden="true"></div>
    {/if}
  </div>

  <!-- Printed inner edge. -->
  <div class="pointer-events-none absolute inset-0 rounded-lg ring-1 ring-inset ring-black/20" aria-hidden="true"></div>

  {#if power !== null && toughness !== null}
    <!-- Numerals in a serif oval: an object on the card, not a UI chip. -->
    <div
      class={`oval pointer-events-none absolute bottom-1 right-1 font-serif font-bold ${size === 'normal' ? 'px-2 py-0.5 text-[12px]' : 'px-1.5 py-px text-[9px]'} ${counters > 0 ? 'oval-brass' : ''}`}
    >
      {power}/{toughness}
    </div>
  {/if}

  {#if tallies > 0}
    <!-- Damage is tallied, not typeset: one red stroke per point. -->
    <div
      class={`tally pointer-events-none absolute right-2 ${size === 'normal' ? 'bottom-8' : 'bottom-6'}`}
      aria-hidden="true"
    >
      {#each Array(tallies) as _, index}
        <i class={index % 2 === 0 ? 'odd' : ''}></i>
      {/each}
    </div>
  {/if}

  {#if counters > 0}
    <!-- Counted in brass: the bead carries the why; the ink stays ivory. -->
    <div class="bead pointer-events-none absolute right-1.5 top-1.5" aria-hidden="true">
      +{counters}/+{counters}
    </div>
  {/if}

  {#if focused}
    <!-- The printer's marks: registered for the press. -->
    <span class="tm tm-tl" aria-hidden="true"></span>
    <span class="tm tm-tr" aria-hidden="true"></span>
    <span class="tm tm-bl" aria-hidden="true"></span>
    <span class="tm tm-br" aria-hidden="true"></span>
  {/if}
</button>

<style>
  .spent {
    filter: grayscale(1) contrast(0.88) brightness(1.02);
    opacity: 0.94;
  }

  .oval {
    background: linear-gradient(to bottom, rgb(24 30 42 / 0.95), rgb(8 11 17 / 0.95));
    border: 1px solid rgb(255 255 255 / 0.22);
    border-radius: 999px;
    color: #faf8f5;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.4);
  }
  .oval-brass {
    border-color: #c3a568;
  }

  .tally {
    display: flex;
    gap: 3px;
  }
  .tally i {
    display: block;
    width: 2px;
    height: 10px;
    border-radius: 1px;
    background: #c14d35;
    box-shadow: 0 0 1px rgb(0 0 0 / 0.5);
    transform: rotate(-7deg) translateY(1px);
  }
  .tally i.odd {
    transform: rotate(9deg);
  }

  .bead {
    background: radial-gradient(circle at 34% 28%, #f0dca8, #c3a568 58%, #8a6c33);
    border: 1px solid rgb(46 34 14 / 0.6);
    border-radius: 999px;
    color: #2c2110;
    font-family: var(--font-serif);
    font-size: 8.5px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.02em;
    padding: 2px 6px;
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.45), inset 0 1px 1px rgb(255 255 255 / 0.5);
    text-shadow: 0 1px 0 rgb(255 246 224 / 0.4);
  }

  .tm {
    position: absolute;
    width: 11px;
    height: 11px;
    z-index: 4;
    pointer-events: none;
  }
  .tm-tl { top: -6px; left: -6px; border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
  .tm-tr { top: -6px; right: -6px; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
  .tm-bl { bottom: -6px; left: -6px; border-bottom: 2px solid var(--accent); border-left: 2px solid var(--accent); }
  .tm-br { bottom: -6px; right: -6px; border-bottom: 2px solid var(--accent); border-right: 2px solid var(--accent); }
  @media (prefers-reduced-motion: no-preference) {
    .tm {
      animation: register 2.4s ease-in-out infinite;
    }
    @keyframes register {
      0%, 100% { transform: translate(0, 0); }
      50% { transform: translate(var(--reg-x, 0), var(--reg-y, 0)); }
    }
    .tm-tl { --reg-x: 2px; --reg-y: 2px; }
    .tm-tr { --reg-x: -2px; --reg-y: 2px; }
    .tm-bl { --reg-x: 2px; --reg-y: -2px; }
    .tm-br { --reg-x: -2px; --reg-y: -2px; }
  }
</style>
