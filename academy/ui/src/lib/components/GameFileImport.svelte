<script lang="ts">
  import {recordedGames} from '$lib/game-file';
  import {recordFetch} from '$lib/records';
  let {onOpen, disabled = false}: {onOpen:(id:string) => void; disabled?:boolean} = $props();
  let games = $state<string[]>([]);
  let selected = $state(0);
  let busy = $state(false);
  let error = $state('');

  async function openSelected(): Promise<void> {
    busy = true; error = '';
    try {
      // Send the original source span: JSON re-encoding can round u64 identities.
      const response = await recordFetch('/api/watch/import', {method:'POST', headers:{'content-type':'application/json'}, body:games[selected]});
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail.replaceAll('_',' ') : result.detail?.message ?? 'Game could not be imported.');
      if (result.token) sessionStorage.setItem(`etude.watch.${result.id}`, result.token);
      onOpen(result.id);
    } catch (reason) { error = reason instanceof Error ? reason.message : String(reason); }
    finally { busy = false; }
  }
  async function importFile(event: Event): Promise<void> {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    busy = true; error = ''; games = [];
    try {
      const source = file.name.endsWith('.gz') ? await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text() : await file.text();
      try { games = recordedGames(source); }
      catch { throw new Error(`${file.name} is not a game file: expected JSON or JSON Lines game records, such as an evaluation trace or an exported .game.jsonl.gz.`); }
      selected = 0;
    } catch (reason) { error = reason instanceof Error ? reason.message : String(reason); }
    finally { busy = false; }
    if (games.length === 1) await openSelected();
  }
</script>
<details class="my-4 rounded border border-line bg-panel p-3">
  <summary class="cursor-pointer">Open a recorded game file</summary>
  <fieldset disabled={disabled || busy} class="mt-3 space-y-2">
    <legend class="sr-only">Import a recording</legend>
    <label for="game-file">Open a game file</label>
    <input id="game-file" aria-describedby="game-file-caption" type="file" accept=".json,.jsonl,.gz" onchange={event => void importFile(event)} />
    <p id="game-file-caption" class="type-caption">An exported training game or an evaluation trace</p>
    {#if games.length > 1}
      <label>Recorded game <select bind:value={selected}>{#each games as _, index}<option value={index}>Game {index + 1}</option>{/each}</select></label>
      <button class="btn btn-secondary" onclick={() => void openSelected()}>Open selected game</button>
    {/if}
  </fieldset>
  {#if error}<p role="alert">{error}</p>{/if}
</details>
