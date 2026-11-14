<script lang="ts">
  import type { PlayerIdentity } from '$lib/player-identity';
  let { identity }: { identity: PlayerIdentity } = $props();
</script>

<span class="player-name" title={identity.description}>
  <span class="sr-only">{identity.description}</span>
  <span class="truncate" aria-hidden="true">{identity.compact}</span>
  {#if identity.shortDigest}<span class="type-rubric shrink-0" aria-hidden="true">· {identity.shortDigest}</span>{/if}
</span>

<style>
  .player-name { display: inline-flex; align-items: baseline; gap: .5rem; min-width: 0; max-width: 100%; vertical-align: bottom; white-space: nowrap; }
</style>
