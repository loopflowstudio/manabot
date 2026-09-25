<script lang="ts">
  import { recordFetch } from '$lib/records';

  let { attemptId }: { attemptId: string } = $props();
  let note = $state('');
  let status = $state('');
  let saving = $state(false);
  let requestId = crypto.randomUUID();

  async function save(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    saving = true;
    status = '';
    try {
      const response = await recordFetch(`/api/traces/${attemptId}/feedback`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ request_id: requestId, note }),
      });
      if (!response.ok) throw new Error('Could not save feedback. Your note is still here; try again.');
      status = 'Feedback saved.';
    } catch (error) {
      status = error instanceof Error ? error.message : 'Feedback could not be saved.';
    } finally {
      saving = false;
    }
  }
</script>

<form onsubmit={save} class="mx-auto my-4 flex max-w-3xl flex-wrap items-end gap-3 rounded border border-line p-4">
  <label class="flex grow flex-col gap-2">
    <span>How did this game feel? <span class="text-ink-2">(optional)</span></span>
    <textarea bind:value={note} maxlength={4000} rows="2" class="rounded border border-line bg-field p-2"
      oninput={() => { requestId = crypto.randomUUID(); status = ''; }}></textarea>
  </label>
  <button type="submit" disabled={saving || !note.trim()} class="rounded border border-line px-3 py-2">Save note</button>
  <a href={`/replay?trace=${encodeURIComponent(attemptId)}`} class="underline">Open replay</a>
  <p role="status" class="w-full">{status}</p>
</form>
