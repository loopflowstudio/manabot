<script lang="ts">
  import { gameStore } from '$lib/game.svelte';

  let enabled = $state(gameStore.deckImport !== null);
  let hero = $state(gameStore.deckImport?.hero.source ?? '');
  let villain = $state(gameStore.deckImport?.villain.source ?? '');
  let heroFilename = $state(gameStore.deckImport?.hero.filename ?? 'hero.txt');
  let villainFilename = $state(gameStore.deckImport?.villain.filename ?? 'villain.txt');
  let packKey = $state(gameStore.deckImport?.pack_key ?? 'tla-pool-v1');
  let messages = $state<string[]>([]);
  let checking = $state(false);
  let revision = 0;

  function update(): void {
    revision += 1;
    messages = [];
    gameStore.deckImport = enabled ? {
      pack_key: packKey, ruleset: 'development-v1',
      hero: { source: hero, filename: heroFilename },
      villain: { source: villain, filename: villainFilename },
    } : null;
  }

  async function loadFile(event: Event, seat: 'hero' | 'villain'): Promise<void> {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    if (file.size > 65536) {
      messages = ['Decklist exceeds the 64 KiB import limit.'];
      return;
    }
    try {
      const source = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer());
      if (seat === 'hero') { hero = source; heroFilename = file.name; }
      else { villain = source; villainFilename = file.name; }
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
        messages = Object.entries(result.documents as Record<string, { issues: { line: number | null; message: string }[] }>).
          flatMap(([seat, document]) => document.issues.map(issue => `${seat === 'hero' ? 'Your deck' : 'Opponent deck'}${issue.line ? `, line ${issue.line}` : ''}: ${issue.message}`));
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
    <label class="mt-3 block text-sm">Content pool <input class="ml-2 rounded border border-line p-2" bind:value={packKey} oninput={update} /></label>
    <p class="mt-2 text-sm text-ink-2">One “count card name” per line. Optional Main and Sideboard headings; # starts a full-line comment.</p>
    <div class="mt-3 grid gap-4 md:grid-cols-2">
      <div>
        <label class="block text-sm" for="hero-deck-file">Your deck file</label>
        <input id="hero-deck-file" type="file" accept=".txt,.dek,text/plain" onchange={event => loadFile(event, 'hero')} />
        <label class="mt-2 block text-sm" for="hero-deck-source">Your deck source</label>
        <textarea id="hero-deck-source" class="w-full rounded border border-line bg-panel p-2 font-mono text-sm" rows="9" bind:value={hero} oninput={update}></textarea>
      </div>
      <div>
        <label class="block text-sm" for="villain-deck-file">Opponent deck file</label>
        <input id="villain-deck-file" type="file" accept=".txt,.dek,text/plain" onchange={event => loadFile(event, 'villain')} />
        <label class="mt-2 block text-sm" for="villain-deck-source">Opponent deck source</label>
        <textarea id="villain-deck-source" class="w-full rounded border border-line bg-panel p-2 font-mono text-sm" rows="9" bind:value={villain} oninput={update}></textarea>
      </div>
    </div>
    <button class="btn mt-3" disabled={checking} onclick={validate}>{checking ? 'Checking…' : 'Check decklists'}</button>
    <div aria-live="polite" class="mt-2 text-sm">{#each messages as message}<p>{message}</p>{/each}</div>
  {/if}
</details>
