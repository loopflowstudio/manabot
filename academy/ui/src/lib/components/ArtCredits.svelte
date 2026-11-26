<script lang="ts">
  /** Who made the card art on screen, and the notice the Fan Content Policy asks for. */
  import { onMount } from 'svelte';
  import { artCredits, type ArtCredit } from '$lib/card-art';

  let credits = $state<[string, ArtCredit][]>([]);
  onMount(() => {
    let active = true;
    let revision = 0;
    const update = (): void => {
      const current = ++revision;
      void artCredits().then(read => { if (active && current === revision) credits = read; }).catch(() => {});
    };
    update();
    window.addEventListener('academy-art-loaded', update);
    return () => { active = false; window.removeEventListener('academy-art-loaded', update); };
  });
</script>

<details class="type-caption text-ink-2" data-testid="art-credits">
  <summary class="cursor-pointer">Card art credits</summary>
  <p class="mt-2">
    Academy is unofficial Fan Content permitted under the
    <a class="underline" href="https://company.wizards.com/en/legal/fancontentpolicy" rel="external noopener">Wizards of the Coast Fan Content Policy</a>.
    Card art and mana symbols are © Wizards of the Coast LLC. Not approved or endorsed by Wizards.
    Images provided by <a class="underline" href="https://scryfall.com" rel="external noopener">Scryfall</a>.
  </p>
  {#if credits.length}
    <ul class="mt-2 grid gap-x-6 gap-y-1 sm:grid-cols-2">
      {#each credits as [name, credit] (name)}
        <li><a class="underline" href={credit.source} rel="external noopener">{name}</a> — {credit.artist}</li>
      {/each}
    </ul>
  {/if}
</details>
