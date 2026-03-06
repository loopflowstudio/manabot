<script lang="ts">
  import { scryfallImageUrl } from '$lib/scryfall';

  export let name: string;
  export let size: 'small' | 'normal' = 'small';
  export let alt = name;
  export let className = '';

  let failed = false;
  $: src = name ? scryfallImageUrl(name, size) : null;
  $: if (src) {
    failed = false;
  }
</script>

{#if src && !failed}
  <img
    src={src}
    alt={alt}
    class={className}
    loading="lazy"
    draggable="false"
    on:error={() => {
      failed = true;
    }}
  />
{:else}
  <slot />
{/if}
