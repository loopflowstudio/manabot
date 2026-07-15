<script lang="ts">
  import type { Snippet } from 'svelte';

  import { CURATED_PACK, resolveTreatment, treatmentBackground } from '$lib/curated-pack';

  interface Props {
    name: string;
    size?: 'small' | 'normal';
    alt?: string;
    className?: string;
    children?: Snippet;
  }

  let { name, size = 'small', alt = undefined, className = '', children = undefined }: Props = $props();

  const treatment = $derived(resolveTreatment(name));
  const background = $derived(treatmentBackground(treatment));
</script>

{#if name}
  <div
    role="img"
    aria-label={alt ?? name}
    data-testid="card-treatment"
    data-card-name={name}
    data-asset-source={treatment.source}
    data-pack-id={treatment.source === 'pack' ? CURATED_PACK.pack.id : undefined}
    class={`relative isolate overflow-hidden ${className}`}
    style:background={background}
  >
    <div class="absolute inset-0 bg-[linear-gradient(135deg,transparent_20%,rgba(255,255,255,0.16)_21%,transparent_22%,transparent_70%,rgba(255,255,255,0.1)_71%,transparent_72%)]"></div>
    <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950 via-slate-950/85 to-transparent px-1.5 pb-1.5 pt-7 text-left text-white">
      <div class={`${size === 'normal' ? 'text-sm' : 'text-[9px]'} font-semibold leading-tight drop-shadow`}>{name}</div>
      <div class={`${size === 'normal' ? 'text-[10px]' : 'text-[7px]'} mt-0.5 uppercase tracking-[0.16em] text-white/65`}>{treatment.motif}</div>
    </div>
  </div>
{:else}
  {@render children?.()}
{/if}
