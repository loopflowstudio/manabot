<script lang="ts">
  import { loadArt } from '$lib/card-art';
  let backUrl = $state<string | null>(null);
  $effect(() => {
    let active = true;
    void loadArt('/api/card-art/back.jpg').then(loaded => {
      if (active && loaded) backUrl = '/api/card-art/back.jpg';
    });
    return () => { active = false; };
  });

  const MATERIAL =
    'radial-gradient(circle at 50% 42%, rgb(154 142 190 / 0.22) 0, transparent 46%), ' +
    'linear-gradient(152deg, #2a2535 0%, #3d3652 56%, #262130 100%)';
  const background = $derived(backUrl
    ? `url('${backUrl}') center / cover no-repeat, ${MATERIAL}`
    : MATERIAL);

  interface Props {
    focused?: boolean;
    clickable?: boolean;
    className?: string;
  }

  let { focused = false, clickable = false, className = '' }: Props = $props();
</script>

<div
  data-art-source={backUrl ? 'scryfall' : 'procedural'}
  class={`back aspect-[5/7] w-20 overflow-hidden rounded-lg shadow ${focused ? 'ring-1 ring-action' : ''} ${clickable ? 'cursor-pointer' : ''} ${className}`}
  style:background={background}
></div>

<style>
  .back {
    /* A fixed world keeps a fixed frame: no adaptive tokens inside. */
    border: 1px solid rgb(6 9 15 / 0.55);
  }
</style>
