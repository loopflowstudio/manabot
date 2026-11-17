<script lang="ts">
  import { manaText, pooledManaText } from '$lib/mana';
  import type { PlayerState, RestrictedMana } from '$lib/types';
  let { player, label }: { player: PlayerState; label: string } = $props();
  function spendingText(restricted: RestrictedMana): string {
    if (restricted.except_nonartifact_spells) return 'anything but nonartifact spells';
    const parts = [];
    if (restricted.spells?.length) parts.push(`${restricted.spells.join(' or ')} spells`);
    if (restricted.abilities?.length) parts.push(`abilities of ${restricted.abilities.join(' or ')} sources`);
    return parts.join(' or ');
  }
  // Native `mana` holds unrestricted mana only; restricted mana is counted
  // separately in `lesson_mana`, so the floating total is their sum.
  const total = $derived(player.mana?.map((amount, color) => amount + (player.lesson_mana?.[color] ?? 0)));
  const hasParts = $derived(Boolean(
    player.restricted_mana?.length || player.lesson_mana?.some(Boolean)
    || player.pooled_mana_classes?.length || player.combat_mana,
  ));
</script>

{#if total}
  <details class="type-caption" data-testid="mana-pool">
    <summary id={`mana-pool-${player.id}`}>{label} floating mana: {manaText(total)}</summary>
    <p>
      {#if hasParts}Amounts below are part of that total, not additional mana. {/if}
      Regular mana expires at step end unless a retention effect applies.
    </p>
    <!-- Restricted totals and effect/combat classes overlap; each is a subset. -->
    {#if player.restricted_mana?.length}
      {#each player.restricted_mana as restricted}
        · {manaText(restricted.mana)} only for {spendingText(restricted)}
      {/each}
    {:else if player.lesson_mana?.some(Boolean)}
      · Restricted mana: {manaText(player.lesson_mana)}
    {/if}
    {#if player.pooled_mana_classes?.length}
      {#each player.pooled_mana_classes as pooled}
        · {pooledManaText(pooled)}
      {/each}
    {:else if player.combat_mana}
      · {player.combat_mana} mana expires at end of combat
    {/if}
    {#if player.keeps_red_mana} · Keeps unspent red mana until end of turn{/if}
  </details>
{:else}
  <p class="type-caption text-ink-2">Mana details not recorded.</p>
{/if}
