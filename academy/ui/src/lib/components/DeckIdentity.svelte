<script lang="ts">
  // Deck names lead with their color identity ("UR Lessons"); present those
  // letters as the official mana symbols (bundled locally — see
  // static/mana/NOTICE.md). A visually hidden copy of each letter keeps the
  // rendered name reading identically to the plain string.
  interface Props {
    name: string;
    symbolsOnly?: boolean;
  }

  let { name, symbolsOnly = false }: Props = $props();

  const parsed = $derived(/^([WUBRG]{1,5}) (.+)$/.exec(name));
  const colors = $derived(parsed ? parsed[1].split('') : []);
  const rest = $derived(parsed ? parsed[2] : name);
</script>

{#if colors.length > 0}<span class="inline-flex items-center gap-1"
  ><span class="inline-flex items-center gap-0.5"
    >{#each colors as color}<span class="inline-flex"
        ><img
          src={`/mana/${color}.svg`}
          alt=""
          class="h-[15px] w-[15px] rounded-full shadow-[-1px_1px_0_rgba(0,0,0,0.3)]"
        /><span class="sr-only">{color}</span></span
      >{/each}</span
  >{#if !symbolsOnly}<span>{' '}{rest}</span>{/if}</span
>{:else if !symbolsOnly}{name}{/if}
