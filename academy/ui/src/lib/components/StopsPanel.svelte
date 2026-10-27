<script lang="ts">
  import { STOP_STEPS } from '$lib/stops';
  import type { StopsConfig, StopSide } from '$lib/types';

  interface Props {
    stops: StopsConfig;
    onToggleStop?: (side: StopSide, step: string) => void;
    onStopOnStackChange?: (value: boolean) => void;
    onAutoPassChange?: (value: boolean) => void;
    onReset?: () => void;
  }

  let {
    stops,
    onToggleStop = undefined,
    onStopOnStackChange = undefined,
    onAutoPassChange = undefined,
    onReset = undefined,
  }: Props = $props();

  const activeCount = $derived(stops.my.length + stops.opponent.length);
  const sides: StopSide[] = ['my', 'opponent'];
</script>

<details class="min-w-0" data-testid="stops-panel">
  <summary class="type-title cursor-pointer select-none py-1 text-display">
    Stops
    <span class="type-label ml-2 rounded-full bg-panel-muted px-2 py-1 text-ink-2">
      {stops.auto_pass ? `${activeCount} set` : 'off'}
    </span>
  </summary>

  <div class="space-y-3 pt-3">
    <label class="flex items-center gap-2 text-ink-2">
      <input
        type="checkbox"
        data-testid="auto-pass-toggle"
        checked={stops.auto_pass}
        onchange={(event) => onAutoPassChange?.(event.currentTarget.checked)}
      />
      Auto-pass priority (MTGO stops)
    </label>

    <div class={stops.auto_pass ? '' : 'pointer-events-none opacity-50'}>
      <div class="grid grid-cols-[minmax(0,1fr)_3rem_3rem] items-center gap-y-1">
        <span></span>
        <span class="type-rubric text-center text-ink-2">Mine</span>
        <span class="type-rubric text-center text-ink-2">Opp</span>
        {#each STOP_STEPS as step}
          <span class="text-ink-2">{step.label}</span>
          {#each sides as side}
            <span class="text-center">
              <input
                type="checkbox"
                data-testid={`stop-${side}-${step.key}`}
                aria-label={`${side === 'my' ? 'My' : 'Opponent'} ${step.label} stop`}
                checked={stops[side].includes(step.key)}
                disabled={!stops.auto_pass}
                onchange={() => onToggleStop?.(side, step.key)}
              />
            </span>
          {/each}
        {/each}
      </div>

      <label class="mt-3 flex items-center gap-2 text-ink-2">
        <input
          type="checkbox"
          data-testid="stop-on-stack"
          checked={stops.stop_on_stack}
          disabled={!stops.auto_pass}
          onchange={(event) => onStopOnStackChange?.(event.currentTarget.checked)}
        />
        Always stop when the stack is not empty
      </label>
    </div>

    <button
      class="type-label inline-block py-1 text-ink-2 underline underline-offset-2 hover:text-ink"
      data-testid="stops-reset"
      onclick={() => onReset?.()}
    >
      Reset to defaults
    </button>
  </div>
</details>
