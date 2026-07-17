<script lang="ts">
  import type { Snippet } from 'svelte';

  import { CURATED_PACK, resolveTreatment, treatmentBackground } from '$lib/curated-pack';

  // The card is a printed plate. Art palettes are fixed dark colors, so
  // everything on the art — grain, varnish, hatching, the name plate — is
  // literal, never adaptive. States are things that happen to printed
  // matter: still wet (sick), reproduced as an etching (spent), turned and
  // re-set for landscape (tapped).
  interface Props {
    name: string;
    size?: 'small' | 'normal';
    landscape?: boolean;
    sick?: boolean;
    spent?: boolean;
    alt?: string;
    className?: string;
    children?: Snippet;
  }

  let {
    name,
    size = 'small',
    landscape = false,
    sick = false,
    spent = false,
    alt = undefined,
    className = '',
    children = undefined,
  }: Props = $props();

  const treatment = $derived(resolveTreatment(name));

  // Real art layers over the procedural treatment. The cache is known to
  // the bundler (import.meta.glob), so a missing file is a lookup miss —
  // never a 404 request. See scripts/fetch-card-art.mjs.
  const ART_URLS = import.meta.glob('/src/lib/card-art/*.jpg', {
    eager: true,
    query: '?url',
    import: 'default',
  }) as Record<string, string>;
  function artSlug(value: string): string {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }
  const background = $derived.by(() => {
    const art = ART_URLS[`/src/lib/card-art/${artSlug(name)}.jpg`];
    const treatmentLayer = treatmentBackground(treatment);
    return art ? `url('${art}') center / cover no-repeat, ${treatmentLayer}` : treatmentLayer;
  });
</script>

{#if name}
  <div
    role="img"
    aria-label={alt ?? name}
    data-testid="card-treatment"
    data-card-name={name}
    data-asset-source={treatment.source}
    data-pack-id={treatment.source === 'pack' ? CURATED_PACK.pack.id : undefined}
    class={`plate-root relative isolate overflow-hidden ${className}`}
    style:background={background}
  >
    <div class="grain" aria-hidden="true"></div>
    {#if !spent}
      <div class="varnish" aria-hidden="true"></div>
    {/if}
    {#if sick}
      <div class="hatch" aria-hidden="true"></div>
    {/if}
    {#if spent}
      <div class="etch" aria-hidden="true"></div>
    {/if}
    <div class={`plate ${landscape ? 'plate-landscape' : ''} ${size === 'normal' ? 'plate-normal' : 'plate-small'}`}>
      <div class="plate-rule" aria-hidden="true"></div>
      <div class="plate-name">{name}</div>
      {#if size === 'normal'}
        <div class="plate-motif">{treatment.motif}</div>
      {/if}
    </div>
  </div>
{:else}
  {@render children?.()}
{/if}

<style>
  /* Print grain: the fixed gradients read as pigment on stock. */
  .grain {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 140px 140px;
    opacity: 0.14;
    mix-blend-mode: overlay;
  }

  /* Varnish light above, vignette below: the card is laminated, the page
     it sits on is not. An etching (spent) is never varnished. */
  .varnish {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
      linear-gradient(to bottom, rgb(255 255 255 / 0.1), transparent 15%),
      radial-gradient(130% 95% at 50% 32%, transparent 58%, rgb(4 8 14 / 0.3));
  }

  /* The ink hasn't dried: engraver's hatching until the first turn. */
  .hatch {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
      135deg,
      rgb(250 248 245 / 0.11) 0 1px,
      transparent 1px 4px
    );
  }

  /* Reproduced as an etching: monochrome hatch letting the paper through. */
  .etch {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      rgb(235 223 198 / 0.15) 0 1px,
      transparent 1px 3px
    );
    background-color: rgb(235 223 198 / 0.08);
  }

  .plate {
    position: absolute;
    inset: auto 0 0 0;
    background: linear-gradient(
      to top,
      rgb(10 13 20 / 0.92) 0%,
      rgb(10 13 20 / 0.78) 58%,
      transparent 100%
    );
    color: #faf8f5;
    text-align: left;
  }
  .plate-small {
    padding: 14px 16px 3px 4px;
  }
  .plate-normal {
    padding: 24px 44px 7px 9px;
  }
  /* Turned and re-set: on a tapped card the plate is typeset for the
     landscape long edge, so a tapped board never reads sideways. */
  .plate-landscape.plate-small {
    padding: 10px 16px 3px 4px;
  }
  .plate-landscape.plate-normal {
    padding: 14px 44px 6px 9px;
  }

  /* The engraved rule: brightest at the left, fading as it runs. */
  .plate-rule {
    height: 1px;
    background: linear-gradient(90deg, rgb(250 248 245 / 0.32), rgb(250 248 245 / 0.05));
    margin-bottom: 3px;
  }
  .plate-small .plate-rule {
    display: none;
  }

  .plate-name {
    font-weight: 600;
    font-size: 7.5px;
    line-height: 1.25;
    letter-spacing: 0.01em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-shadow: 0 1px 2px rgb(0 0 0 / 0.45);
  }
  .plate-normal .plate-name {
    font-family: var(--font-serif);
    font-size: 13px;
    letter-spacing: 0.012em;
  }
  .plate-motif {
    margin-top: 3px;
    font-family: var(--font-mono);
    font-size: 7px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgb(250 248 245 / 0.62);
    white-space: nowrap;
  }
</style>
