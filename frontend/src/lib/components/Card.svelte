<script lang="ts">
  import CardImage from './CardImage.svelte';

  export let name: string;
  export let power: number | null = null;
  export let toughness: number | null = null;
  export let focused = false;
  export let clickable = false;
  export let tapped = false;
  export let dimmed = false;
  export let damage = 0;
  export let size: 'small' | 'normal' = 'small';
  export let onSelect: (() => void) | undefined = undefined;
  export let onHoverStart: (() => void) | undefined = undefined;
  export let onHoverEnd: (() => void) | undefined = undefined;

  $: widthClass = size === 'normal' ? 'w-40' : 'w-20';
  $: imageClass = 'h-full w-full rounded-md object-cover';
</script>

<button
  type="button"
  aria-label={name}
  class={`group relative aspect-[5/7] ${widthClass} overflow-visible rounded-lg border bg-slate-900 text-left shadow transition ${focused ? 'border-blue-400 ring-1 ring-blue-400/70' : 'border-slate-700'} ${clickable ? 'cursor-pointer hover:-translate-y-1 hover:border-amber-300' : 'cursor-default'} ${tapped ? 'rotate-90' : ''} ${dimmed ? 'opacity-70' : ''}`}
  on:click={() => {
    if (clickable) {
      onSelect?.();
    }
  }}
  on:mouseenter={() => onHoverStart?.()}
  on:mouseleave={() => onHoverEnd?.()}
>
  <div class="absolute inset-0 overflow-hidden rounded-lg">
    <CardImage name={name} size={size} alt={name} className={imageClass}>
      <div class="flex h-full w-full flex-col justify-between bg-slate-800 p-2 text-left text-[11px] text-slate-200">
        <div class="font-semibold leading-tight">{name}</div>
        {#if power !== null && toughness !== null}
          <div class="self-end rounded bg-slate-950/80 px-1.5 py-0.5 text-[10px] font-semibold">
            {power}/{toughness}
          </div>
        {/if}
      </div>
    </CardImage>
  </div>

  <div class="pointer-events-none absolute inset-0 rounded-lg ring-1 ring-inset ring-black/20"></div>

  {#if power !== null && toughness !== null}
    <div class="pointer-events-none absolute bottom-1 right-1 rounded bg-slate-950/90 px-1.5 py-0.5 text-[10px] font-semibold text-slate-100">
      {power}/{toughness}
    </div>
  {/if}

  {#if damage > 0}
    <div class="pointer-events-none absolute right-1 top-1 rounded-full bg-rose-600 px-2 py-0.5 text-[10px] font-semibold text-white">
      {damage}
    </div>
  {/if}
</button>
