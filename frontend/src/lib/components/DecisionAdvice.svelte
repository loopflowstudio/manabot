<script lang="ts">
  import { onMount } from 'svelte';

  import type { AdviceScenarioSummary } from '$lib/advice';
  import type { DecisionEvidence } from '$lib/study-protocol';
  import type { ExperienceFrame, InteractionOffer } from '$lib/types';

  interface Props {
    mode: 'live' | 'study';
    scenarios: AdviceScenarioSummary[];
    selectedScenarioId: string;
    frame: ExperienceFrame | null;
    offers: InteractionOffer[];
    evidence: DecisionEvidence | null;
    deltas: Record<string, Record<string, number>> | null;
    status: 'ok' | 'unavailable' | 'loading';
    reason: string | null;
    advisorId: string;
    computeId: string;
    reducedMotion?: boolean;
    onSelectScenario: (id: string) => void;
  }

  let {
    mode,
    scenarios,
    selectedScenarioId,
    frame,
    offers,
    evidence,
    deltas,
    status,
    reason,
    advisorId,
    computeId,
    reducedMotion = false,
    onSelectScenario,
  }: Props = $props();

  // Reduced motion is a client preference; read the media query once on mount
  // and mirror it for the release e2e assertReducedMotion helper. The derived
  // value falls back to the prop before mount (and in SSR, where onMount does
  // not run), so unit tests can drive it without a browser.
  let mediaReducedMotion = $state<boolean | null>(null);
  const effectiveReducedMotion = $derived(mediaReducedMotion ?? reducedMotion);
  onMount(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    mediaReducedMotion = media.matches;
    const apply = (event: MediaQueryListEvent) => {
      mediaReducedMotion = event.matches;
    };
    media.addEventListener('change', apply);
    return () => media.removeEventListener('change', apply);
  });

  const projection = $derived(frame?.projection ?? null);

  function alternativeId(offer: InteractionOffer): string {
    return `offer-${offer.id}`;
  }
  function policyFor(alternative: string): number {
    return evidence?.policy_mass.find((row) => row.alternative === alternative)?.probability ?? 0;
  }
  function valueFor(alternative: string): number | null {
    return evidence?.search_value.find((row) => row.alternative === alternative)?.expected_match_points ?? null;
  }
  function uncertaintyFor(alternative: string): number | null {
    return evidence?.uncertainty.find((row) => row.alternative === alternative)?.standard_error ?? null;
  }
  function favorableFor(alternative: string): { favorable: number; sampled: number } | null {
    const row = evidence?.sampled_world_robustness.find((r) => r.alternative === alternative);
    return row ? { favorable: row.favorable_worlds, sampled: row.sampled_worlds } : null;
  }
  function deltaFor(alternative: string): Record<string, number> | null {
    return deltas?.[alternative] ?? null;
  }
  function signed(value: number): string {
    const rounded = value.toFixed(3);
    return value > 0 ? `+${rounded}` : rounded;
  }
</script>

<details class="min-w-0" data-testid="decision-advice" data-mode={mode} data-reduced-motion={effectiveReducedMotion}>
  <summary class="type-title cursor-pointer select-none py-1 text-display">
    Decision Advice
    <span class="type-label ml-2 rounded-full bg-panel-muted px-2 py-1 text-ink-2">
      {mode === 'live' ? 'live' : 'study'}
    </span>
  </summary>

  <div class="space-y-4 pt-3">
    <!-- Beliefs: the viewer-safe hidden-info scenario selector. -->
    <section data-testid="advice-beliefs" aria-label="Belief scenarios">
      <h3 class="type-rubric mb-1 text-ink-2">Beliefs</h3>
      <fieldset class="m-0 border-0 p-0">
        <legend class="sr-only">Choose a belief about the opponent's hand</legend>
        {#each scenarios as scenario}
          <label
            class="flex cursor-pointer gap-2 py-1 text-ink-2"
            data-testid="advice-scenario-option"
            data-scenario-id={scenario.landmark_id}
          >
            <input
              type="radio"
              name="advice-scenario"
              value={scenario.landmark_id}
              checked={scenario.landmark_id === selectedScenarioId}
              onchange={() => onSelectScenario(scenario.landmark_id)}
            />
            <span class="min-w-0">
              <span class="block text-ink">{scenario.label}</span>
              <span class="type-caption block text-ink-3">{scenario.description}</span>
              <span class="type-caption block text-ink-3">Inferred range: {scenario.inferred_range}</span>
            </span>
          </label>
        {/each}
      </fieldset>
    </section>

    {#if status === 'loading'}
      <p data-testid="advice-loading" class="type-caption text-ink-3">Loading advice…</p>
    {:else if status === 'unavailable'}
      <p
        data-testid="advice-unavailable"
        data-reason={reason ?? 'unknown'}
        role="status"
        class="type-caption text-ink-3"
      >
        Advice unavailable{reason ? ` (${reason})` : ''}. No evidence is shown.
      </p>
    {:else if projection && evidence}
      <!-- Facts: the public board. No opponent hand identities. -->
      <section data-testid="advice-facts" aria-label="Public board facts">
        <h3 class="type-rubric mb-1 text-ink-2">Facts</h3>
        <dl class="grid grid-cols-2 gap-x-4 gap-y-1 type-caption text-ink-2">
          <dt>Turn</dt><dd>{projection.turn.turn_number} · {projection.turn.phase}</dd>
          <dt>Hero life</dt><dd>{projection.agent.life}</dd>
          <dt>Opponent life</dt><dd>{projection.opponent.life}</dd>
          <dt>Hero hand</dt><dd>{projection.agent.hand.length}</dd>
          <dt>Opponent hand</dt>
          <dd>{projection.opponent.hand_hidden_count ?? projection.opponent.hand.length} (hidden)</dd>
          <dt>Hero board</dt><dd>{projection.agent.battlefield.length}</dd>
          <dt>Opponent board</dt><dd>{projection.opponent.battlefield.length}</dd>
        </dl>
      </section>

      <!-- Advice: per-action policy mass, expected match points, uncertainty. -->
      <section data-testid="advice-advice" aria-label="Advisor evidence">
        <h3 class="type-rubric mb-1 text-ink-2">Advice</h3>
        <ul class="m-0 list-none space-y-2 p-0">
          {#each offers as offer}
            {@const alt = alternativeId(offer)}
            {@const probability = policyFor(alt)}
            {@const value = valueFor(alt)}
            {@const unc = uncertaintyFor(alt)}
            {@const favorable = favorableFor(alt)}
            <li data-testid="advice-action-row" data-action-id={alt}>
              <div class="flex items-baseline justify-between gap-2">
                <span class="text-ink">{offer.label}</span>
                <span class="type-caption text-ink-3">{(probability * 100).toFixed(0)}%</span>
              </div>
              <div
                class="h-1.5 rounded bg-panel-muted"
                role="progressbar"
                aria-valuenow={probability}
                aria-valuemin={0}
                aria-valuemax={1}
              >
                <div
                  class="h-full rounded bg-forest {effectiveReducedMotion ? '' : 'advice-bar'}"
                  style={`width: ${Math.max(probability * 100, 1)}%`}
                ></div>
              </div>
              <dl class="grid grid-cols-2 gap-x-4 type-caption text-ink-3">
                <dt>Value</dt><dd>{value === null ? '—' : value.toFixed(3)}</dd>
                <dt>Uncertainty</dt><dd>{unc === null ? '—' : `±${unc.toFixed(3)}`}</dd>
                {#if favorable}<dt>Favorable</dt><dd>{favorable.favorable}/{favorable.sampled} worlds</dd>{/if}
              </dl>
            </li>
          {/each}
        </ul>
      </section>

      <!-- Deltas: explicit signed differences vs the other scenario. -->
      <section data-testid="advice-deltas" aria-label="Scenario deltas">
        <h3 class="type-rubric mb-1 text-ink-2">Deltas vs other scenario</h3>
        <ul class="m-0 list-none space-y-1 p-0">
          {#each offers as offer}
            {@const alt = alternativeId(offer)}
            {@const delta = deltaFor(alt)}
            <li class="type-caption text-ink-2" data-testid="advice-delta-row" data-action-id={alt}>
              <span class="text-ink">{offer.label}:</span>
              {#if delta}
                <span class="ml-1">policy {signed(delta.policy_mass)}</span>
                <span class="ml-2">value {signed(delta.search_value)}</span>
                <span class="ml-2">unc {signed(delta.uncertainty)}</span>
              {:else}
                <span class="ml-1 text-ink-3">—</span>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    <footer data-testid="advice-footer" class="type-caption border-t border-line pt-2 text-ink-3">
      Advisor {advisorId} · compute {computeId} · advisory only — submit through the ActionPanel.
    </footer>
  </div>
</details>

<style>
  .advice-bar {
    transition: width 200ms ease-out;
  }
</style>
