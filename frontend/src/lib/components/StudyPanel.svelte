<script lang="ts">
  import { tick } from 'svelte';

  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { StudyDisplayPlan, StudyPlanKind } from '$lib/study-runtime';
  import type { StudyPhase } from '$lib/study.svelte';
  import type { InteractionOffer } from '$lib/types';

  interface Props {
    phase: StudyPhase;
    restored: RestoredReplayDecision;
    plans?: StudyDisplayPlan[];
    selectedPlan?: StudyPlanKind | null;
    busy?: boolean;
    errorMessage?: string | null;
    onRetry?: (offer: InteractionOffer) => void;
    onReveal?: () => void;
    onReturn?: () => void;
    onPreview?: (plan: StudyPlanKind) => void;
    onFocusPlan?: (plan: StudyPlanKind | null) => void;
  }

  let {
    phase,
    restored,
    plans = [],
    selectedPlan = null,
    busy = false,
    errorMessage = null,
    onRetry = undefined,
    onReveal = undefined,
    onReturn = undefined,
    onPreview = undefined,
    onFocusPlan = undefined,
  }: Props = $props();

  let controls: HTMLElement | null = $state(null);
  let previousPhase: StudyPhase | null = null;
  let previousBusy = false;

  $effect(() => {
    const nextPhase = phase;
    const nextBusy = busy;
    if (nextPhase === previousPhase && nextBusy === previousBusy) return;
    previousPhase = nextPhase;
    previousBusy = nextBusy;
    if (nextBusy) return;
    void tick().then(() => {
      if (nextPhase === 'restored') {
        controls?.querySelector<HTMLButtonElement>('[data-testid="study-retry-offer"]')?.focus();
      } else if (nextPhase === 'retried') {
        controls?.querySelector<HTMLButtonElement>('[data-testid="study-reveal"]')?.focus();
      } else if (nextPhase === 'revealed') {
        controls
          ?.querySelector<HTMLButtonElement>('[role="radio"][aria-checked="true"]')
          ?.focus();
      }
    });
  });

  function movePlanFocus(event: KeyboardEvent, index: number): void {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
    const next = (index + direction + plans.length) % plans.length;
    const plan = plans[next];
    controls?.querySelectorAll<HTMLButtonElement>('[role="radio"]')[next]?.focus();
    onFocusPlan?.(plan.kind);
    onPreview?.(plan.kind);
  }

  function activatePlan(event: KeyboardEvent, kind: StudyPlanKind): void {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    onPreview?.(kind);
  }

  function formatPercent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }
</script>

<section
  bind:this={controls}
  data-testid="study-panel"
  aria-labelledby="study-heading"
  class="mt-4 rounded border border-line-strong bg-field px-4 py-4"
>
  <div class="flex flex-wrap items-start justify-between gap-3">
    <div>
      <p class="type-rubric text-mountain-ink">Study · Decision {restored.ordinal + 1}</p>
      <h2 id="study-heading" class="type-title mt-1 text-display">
        {phase === 'restored' ? 'Retry before reveal' : phase === 'retried' ? 'Your line is committed' : 'Compare plans'}
      </h2>
    </div>
    <button
      type="button"
      data-testid="study-return"
      class="btn btn-secondary min-h-11"
      onclick={() => onReturn?.()}
      disabled={busy}
    >
      Return to score
    </button>
  </div>

  {#if errorMessage}
    <p
      role="status"
      aria-live="polite"
      data-testid="study-error"
      class="type-caption mt-3 border-l-2 border-mountain pl-3 text-mountain-ink"
    >
      {errorMessage}
    </p>
  {/if}

  {#if phase === 'restored'}
    <p class="type-annotation mt-3 text-ink-2">
      Choose your line from the exact historical offer. Policy and Search stay sealed until one command is accepted.
    </p>
    <div class="mt-3 grid gap-2" role="group" aria-label="Retry choices">
      {#each restored.frame.offers as offer}
        <button
          type="button"
          data-testid="study-retry-offer"
          data-offer-id={offer.id}
          class="min-h-11 rounded border border-line-strong bg-panel px-3 py-2 text-left font-medium text-ink hover:border-action focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-action"
          onclick={() => onRetry?.(offer)}
          disabled={busy}
        >
          {offer.label}
        </button>
      {/each}
    </div>
  {:else if phase === 'retried'}
    <p class="type-annotation mt-3 text-ink-2">
      Your prediction is locked. Reveal asks the evidence provider now; returning leaves it sealed.
    </p>
    <button
      type="button"
      data-testid="study-reveal"
      class="btn btn-primary mt-3 min-h-11 w-full sm:w-auto"
      onclick={() => onReveal?.()}
      disabled={busy}
    >
      {busy ? 'Checking evidence…' : 'Reveal plans'}
    </button>
  {:else if phase === 'revealed'}
    <p class="type-annotation mt-3 text-ink-2">
      Played, Policy, and Search are separate judgments. Focus a plan to highlight its authoritative table objects; select it to play its bounded continuation.
    </p>
    <div
      class="mt-3 grid gap-3 lg:grid-cols-3"
      role="radiogroup"
      aria-label="Study plans"
      tabindex="-1"
      onkeydown={(event) => {
        if (event.key === 'Escape') onReturn?.();
      }}
    >
      {#each plans as plan, index}
        <button
          type="button"
          role="radio"
          data-testid={`study-plan-${plan.kind}`}
          aria-checked={selectedPlan === plan.kind}
          tabindex={selectedPlan === plan.kind || (selectedPlan === null && index === 0) ? 0 : -1}
          class={`min-h-11 rounded border px-4 py-3 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-action ${selectedPlan === plan.kind ? 'border-action bg-panel shadow-sheet' : 'border-line-strong bg-field hover:border-action'}`}
          onmouseenter={() => onFocusPlan?.(plan.kind)}
          onmouseleave={() => onFocusPlan?.(null)}
          onfocus={() => onFocusPlan?.(plan.kind)}
          onblur={() => onFocusPlan?.(null)}
          onclick={() => onPreview?.(plan.kind)}
          onkeydown={(event) => {
            movePlanFocus(event, index);
            activatePlan(event, plan.kind);
          }}
          disabled={busy}
        >
          <span class="type-title block text-display">{plan.label}</span>
          <span class="mt-1 block font-medium text-ink">{plan.offer.label}</span>
          {#if plan.kind !== 'played' && plan.sameAsPlayed}
            <span class="type-caption mt-1 block text-ink-2">Same line as played</span>
          {/if}
          {#if plan.policyProbability !== null}
            <span class="type-caption mt-2 block text-ink-2">Policy probability · {formatPercent(plan.policyProbability)}</span>
          {/if}
          {#if plan.expectedMatchPoints !== null}
            <span class="type-caption mt-1 block text-ink-2">Search value · {plan.expectedMatchPoints.toFixed(3)} match points</span>
          {/if}
          {#if plan.visits !== null}
            <span class="type-caption mt-1 block text-ink-2">Visits · {plan.visits}</span>
          {/if}
          {#if plan.favorableWorlds !== null && plan.sampledWorlds !== null}
            <span class="type-caption mt-1 block text-ink-2">Robustness · {plan.favorableWorlds}/{plan.sampledWorlds} worlds</span>
          {/if}
          {#if plan.standardError !== null}
            <span class="type-caption mt-1 block text-ink-2">Uncertainty · ±{plan.standardError.toFixed(3)} ({plan.uncertaintyMethod})</span>
          {/if}
        </button>
      {/each}
    </div>
    <p class="sr-only" aria-live="polite">
      {selectedPlan ? `${selectedPlan} plan selected` : 'No plan selected'}
    </p>
  {/if}
</section>
