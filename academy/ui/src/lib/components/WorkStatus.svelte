<script lang="ts">
  import type { WorkStatus } from '../types';
  let { work, connected = true }: { work: WorkStatus | null; connected?: boolean } = $props();
  let elapsed = $state(0);
  const label = $derived(work ? ({load: 'Loading bot', choose: 'Choosing bot action', search: 'Searching', advance: 'Advancing live game'}[work.kind]) : '');
  $effect(() => {
    if (!work || work.state !== 'running' || !connected) return;
    const started = work.started_at;
    const update = () => elapsed = Math.max(0, Math.floor(Date.now() / 1000 - started));
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  });
</script>

{#if work?.state === 'running' && connected}
  <p class="type-caption py-2 text-ink-2" data-testid="work-status">
    <span role="status">{label}{work.chooser !== null ? ` · Seat ${work.chooser + 1}` : ''}</span>
    {#if elapsed >= 1}<span aria-hidden="true"> · {elapsed} s elapsed</span>{/if}
  </p>
{/if}
