<script lang="ts">
  import { STOP_STEPS } from '$lib/stops';
  import type { StopsConfig, StopSide } from '$lib/types';

  interface Props {
    stops: StopsConfig;
    open?: boolean;
    element?: HTMLDetailsElement | null;
    onToggleStop?: (side: StopSide, step: string) => void;
    onStopOnStackChange?: (value: boolean) => void;
    onAutoPassChange?: (value: boolean) => void;
    onReset?: () => void;
  }

  let {
    stops,
    open = $bindable(false),
    element = $bindable(null),
    onToggleStop = undefined,
    onStopOnStackChange = undefined,
    onAutoPassChange = undefined,
    onReset = undefined,
  }: Props = $props();

  const activeCount = $derived(stops.my.length + stops.opponent.length);
  const sides: StopSide[] = ['my', 'opponent'];
</script>

<details bind:open bind:this={element} class="min-w-0" data-testid="stops-panel">
  <summary class="type-title cursor-pointer select-none py-1 text-display">
    Stops
    <span class="type-label ml-2 rounded-full bg-panel-muted px-2 py-1 text-ink-2">
      {stops.auto_pass ? `${activeCount} set` : 'off'}
    </span>
  </summary>

  <div class="space-y-3 pt-3">
    <p class="type-caption text-ink-2">Checked steps hold priority when you have a choice. Unchecked steps pass automatically unless a stack response requires a stop. Changes may advance play immediately. Target, payment and combat choices still appear.</p>
    <label class="flex items-center gap-2 text-ink-2">
      <input
        type="checkbox"
        data-testid="auto-pass-toggle"
        checked={stops.auto_pass}
        onchange={(event) => onAutoPassChange?.(event.currentTarget.checked)}
      />
      Use priority stops
    </label>

    <div class={stops.auto_pass ? '' : 'pointer-events-none opacity-50'}>
      <div class="grid grid-cols-[minmax(0,1fr)_5rem_6rem] items-center gap-y-1">
        <span></span>
        <span class="type-rubric text-center text-ink-2">Your turn</span>
        <span class="type-rubric text-center text-ink-2">Opponent turn</span>
        {#each STOP_STEPS as step}
          <span class="text-ink-2">{step.label}</span>
          {#each sides as side}
            <span class="text-center">
              <input
                type="checkbox"
                data-testid={`stop-${side}-${step.key}`}
                aria-label={`${side === 'my' ? 'Your turn' : 'Opponent turn'} ${step.label} stop`}
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
        Stop when the stack offers a response choice
      </label>
    </div>

    <p class="type-caption text-ink-2">Pass Turn (F6) yields through checked stops and stack responses until the turn ends or a non-priority choice needs you. With priority stops off, every priority choice still appears.</p>
    <button
      class="type-label inline-block py-1 text-ink-2 underline underline-offset-2 hover:text-ink"
      data-testid="stops-reset"
      onclick={() => onReset?.()}
    >
      Reset to defaults
    </button>
  </div>
</details>
