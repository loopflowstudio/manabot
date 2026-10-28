<script lang="ts">
  import { gameStore } from '$lib/game.svelte';

  const seats = [
    { key: 'hero', label: 'Your deck' },
    { key: 'villain', label: 'Opponent deck' },
  ] as const;
  let enabled = $state(gameStore.deckImport !== null);
  let draft = $state(gameStore.deckImport ?? {
    pack_key: 'tla-pool-v1', ruleset: 'development-v1',
    hero: { source: '', filename: 'hero.txt' },
    villain: { source: '', filename: 'villain.txt' },
  });
  let messages = $state<string[]>([]);
  let checking = $state(false);
  let revision = 0;

  function update(): void {
    revision += 1;
    messages = [];
    gameStore.deckImport = enabled ? draft : null;
  }

  async function loadFile(event: Event, seat: 'hero' | 'villain'): Promise<void> {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    if (file.size > 65536) {
      messages = ['Decklist exceeds the 64 KiB import limit.'];
      return;
    }
    try {
      const source = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
      draft[seat] = { source, filename: file.name };
      update();
    } catch {
      messages = ['Could not read this file as UTF-8 text.'];
    }
  }

  async function validate(): Promise<void> {
    const started = revision;
    checking = true;
    try {
      const response = await fetch('/api/decklists/validate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gameStore.deckImport),
      });
      const result = await response.json();
      if (started !== revision) return;
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Could not validate decklists.');
      if (result.admitted) {
        messages = [`Ready — ${result.ruleset}, ${result.world_version}. Choose New Game to play these decks.`];
      } else {
        const documents = result.documents as Partial<Record<'hero' | 'villain', {
          issues: { line: number | null; message: string }[];
        }>>;
        messages = seats.flatMap(({ key, label }) =>
          (documents[key]?.issues ?? []).map(issue =>
            `${label}${issue.line ? `, line ${issue.line}` : ''}: ${issue.message}`,
          ),
        );
      }
    } catch (error) {
      if (started === revision) messages = [error instanceof Error ? error.message : 'Could not validate decklists.'];
    } finally { checking = false; }
  }
</script>

<details class="mt-4 rounded-lg border border-line bg-panel p-4">
  <summary class="cursor-pointer font-medium">Import decklist files</summary>
  <p class="mt-2 text-sm text-ink-2">Development format — supported cards only, no tournament legality claim. Sideboards are fixed outside-game copies. Checkpoint opponents are not available for imported decks.</p>
  <label class="mt-3 flex items-center gap-2"><input type="checkbox" bind:checked={enabled} onchange={update} /> Use imported decks for the next game</label>
  {#if enabled}
    <label class="mt-3 block text-sm">Content pool <input class="ml-2 rounded border border-line p-2" bind:value={draft.pack_key} oninput={update} /></label>
    <p class="mt-2 text-sm text-ink-2">One “count card name” per line. Optional Main and Sideboard headings; # starts a full-line comment.</p>
    <div class="mt-3 grid gap-4 md:grid-cols-2">
      {#each seats as { key, label }}
        <div>
          <label class="block text-sm" for={`${key}-deck-file`}>{label} file</label>
          <input id={`${key}-deck-file`} type="file" accept=".txt,.dek,text/plain" onchange={event => loadFile(event, key)} />
          <label class="mt-2 block text-sm" for={`${key}-deck-source`}>{label} source</label>
          <textarea id={`${key}-deck-source`} class="w-full rounded border border-line bg-panel p-2 font-mono text-sm" rows="9" bind:value={draft[key].source} oninput={update}></textarea>
        </div>
      {/each}
    </div>
    <button class="btn mt-3" disabled={checking} onclick={validate}>{checking ? 'Checking…' : 'Check decklists'}</button>
    <div aria-live="polite" class="mt-2 text-sm">{#each messages as message}<p>{message}</p>{/each}</div>
  {/if}
</details>
