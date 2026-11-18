<script lang="ts">
  /** A small art crop of a named card, for the stack and the recap; decorative beside its name. */
  import { artUrl, loadArt } from '$lib/card-art';

  let { name }: { name: string } = $props();
  const url = $derived(artUrl(name));
  let loaded = $state<string | null>(null);
  $effect(() => {
    const requested = url;
    let active = true;
    void loadArt(requested, name).then(ok => { if (active && ok) loaded = requested; });
    return () => { active = false; };
  });
</script>

<span class="thumb" aria-hidden="true" style:background-image={loaded === url ? `url('${url}')` : undefined}></span>

<style>
  .thumb {
    flex: none; width: 26px; height: 36px; border-radius: 3px; border: 1px solid rgb(20 14 8 / 0.5);
    background: var(--bg-muted) center 30% / cover no-repeat;
  }
</style>
