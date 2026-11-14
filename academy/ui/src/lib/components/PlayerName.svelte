<script lang="ts">
  import { shortHash, type PlayerIdentity } from '$lib/player-identity';
  // `chip={false}` where the page already shows the digest beside the name.
  let { identity, chip = true }: { identity: PlayerIdentity; chip?: boolean } = $props();
</script>

<span class="player-name" title={identity.description}>
  <span class="sr-only">{identity.description}</span>
  <span class="truncate" aria-hidden="true">{identity.compact}</span>
  {#if chip && identity.digest}<span class="type-rubric shrink-0" aria-hidden="true">· {shortHash(identity.digest)}</span>{/if}
</span>

<style>
  .player-name { display: inline-flex; align-items: baseline; gap: .5rem; min-width: 0; max-width: 100%; vertical-align: bottom; white-space: nowrap; }
</style>
