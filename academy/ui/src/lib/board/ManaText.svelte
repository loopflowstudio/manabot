<script lang="ts">
  /**
   * A line of words that may name mana ("Pay {R} for Firebending Lesson"):
   * each run of braced symbols is drawn as pips, the rest as text. The pips'
   * accessible text is the run itself, so the line reads as it is written.
   */
  import { manaRuns } from '$lib/mana';
  import Pips from './Pips.svelte';

  let { text, size = 14 }: { text: string; size?: number } = $props();
  const runs = $derived(manaRuns(text));
</script>

{#each runs as run, index (index)}{#if run.mana}<span class="mana"><Pips cost={run.text} {size} /></span>{:else}{run.text}{/if}{/each}

<style>
  .mana { display: inline-flex; vertical-align: -0.15em; }
</style>
