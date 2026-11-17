<script lang="ts">
  import type { BotDecision } from '$lib/recording';
  import type { InteractionOffer } from '$lib/types';
  let {recorded = null, chosenOffer, offers}: {recorded?: BotDecision | null; chosenOffer?: number; offers: InteractionOffer[]} = $props();
</script>
<details open class="min-w-0" data-testid="recorded-decision-evidence">
  <summary class="type-title cursor-pointer py-1 text-display">Recorded policy and search evidence</summary>
  <div class="space-y-4 pt-3">
      <p class="type-caption">Recorded at this decision. Chosen offers are marked ✓.</p>
      <p>Value: {recorded?.value?.toFixed(4) ?? 'not retained'} {recorded?.value_semantic ?? ''}</p>
      <p>Update: {recorded?.update ?? 'not retained'}</p>
      {#if recorded?.unavailable}<p class="type-caption text-ink-2">{recorded.unavailable}</p>{/if}
      <table class="w-full text-left type-caption">
        <thead><tr><th>Legal action</th>{#if recorded?.mirror}<th>Raw policy</th><th>Search values</th><th>Updated policy</th>{:else}<th>Policy</th><th>Search score</th>{/if}</tr></thead>
        <tbody>{#each offers as offer, index}
          <tr class:bg-panel-muted={offer.id === chosenOffer}>
            <td class="py-2">{offer.id === chosenOffer ? '✓ ' : ''}{offer.label}</td>
            {#if recorded?.mirror}<td>{recorded.mirror.prior[index] === undefined ? 'not retained' : `${(recorded.mirror.prior[index] * 100).toFixed(2)}%`}</td>{/if}
            {#if !recorded?.mirror}<td>{recorded?.probabilities[index] === undefined ? 'not retained' : `${(recorded.probabilities[index] * 100).toFixed(2)}%`}</td>{/if}
            <td>{recorded?.search_scores[index]?.toFixed(4) ?? 'not retained'}</td>
            {#if recorded?.mirror}<td>{recorded.probabilities[index] === undefined ? 'not retained' : `${(recorded.probabilities[index] * 100).toFixed(2)}%`}</td>{/if}
          </tr>
        {/each}</tbody>
      </table>
      {#if recorded?.belief}
        <p class="type-caption">Belief: {recorded.belief.model} · {recorded.belief.sampled_worlds} sampled worlds · condition mass {recorded.belief.condition_mass.toFixed(4)}</p>
      {/if}
      {#if recorded?.mirror}
        <p class="type-caption">Search: {recorded.mirror.worlds} worlds × depth {recorded.mirror.depth} · {recorded.mirror.rollouts} rollouts · {recorded.mirror.seconds.toFixed(3)} s of {recorded.mirror.decision_seconds} s · α {recorded.mirror.alpha}, β {recorded.mirror.beta}</p>
        <p class="type-caption">Uniform belief (physical deal): hidden cards are sampled with copy counts and public knowledge.</p>
        <details class="type-caption"><summary>Search identity</summary>
          <p>Seed: {recorded.mirror.seed}</p>
          <p class="break-all">Receipt SHA-256: {recorded.mirror.receipt_sha256}</p>
        </details>
      {/if}
      {#if recorded?.factors?.length}
        <details><summary>Recorded compound sequence factors</summary>
          <ol>{#each recorded.factors as factor, index}<li class="py-2 type-caption">Factor {index + 1}: chose {factor.chosen}; probabilities {factor.probabilities.map(p => `${(p * 100).toFixed(2)}%`).join(', ')}; value {factor.value.toFixed(4)}</li>{/each}</ol>
        </details>
      {/if}
  </div>
</details>
