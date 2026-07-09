<script lang="ts">
  import type { Snippet } from 'svelte';

  import { scryfallImageUrl } from '$lib/scryfall';

  interface Props {
    name: string;
    size?: 'small' | 'normal';
    alt?: string;
    className?: string;
    children?: Snippet;
  }

  let { name, size = 'small', alt = undefined, className = '', children = undefined }: Props = $props();

  let failedSrc = $state<string | null>(null);
  const src = $derived(name ? scryfallImageUrl(name, size) : null);
  const failed = $derived(src !== null && failedSrc === src);
</script>

{#if src && !failed}
  <img
    {src}
    alt={alt ?? name}
    class={className}
    loading="lazy"
    draggable="false"
    onerror={() => {
      failedSrc = src;
    }}
  />
{:else}
  {@render children?.()}
{/if}
