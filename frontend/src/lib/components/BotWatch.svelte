<script lang="ts">
  import { onMount } from 'svelte';
  import GameBoard from './GameBoard.svelte';
  import DeckSelector from './DeckSelector.svelte';
  import DecisionAdvice from './DecisionAdvice.svelte';
  import StudyPanel from './StudyPanel.svelte';
  import type { DeckChoice } from '$lib/decks';
  import type { WatchDecision } from '$lib/watch';
  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { InteractionOffer, Observation } from '$lib/types';
  import type { StudyRetryResponse } from '$lib/study-runtime';
  import { rememberRecordCredential, recordFetch } from '$lib/records';

  let players = $state(['random', 'heuristic']);
  let hero = $state<DeckChoice>('ur_lessons');
  let villain = $state<DeckChoice>('gw_allies');
  let seed = $state('42');
  let id = $state('');
  let token = $state('');
  let current = $state<WatchDecision | null>(null);
  let ordinal = $state(0);
  let revision = $state(0);
  let finished = $state(false);
  let count = $state(0);
  let reveal = $state(true);
  let playing = $state(false);
  let speed = $state(1);
  let busy = $state(false);
  let error = $state('');
  let trained = $state(false);
  let ready = $state(false);
  let trainedName = $state('Configured checkpoint');
  let restored = $state<RestoredReplayDecision | null>(null);
  let attempt = $state<string | null>(null);
  let branch = $state<Observation | null>(null);
  let phase = $state<'restored' | 'retried'>('restored');
  let importedGames = $state<unknown[]>([]);
  let selectedGame = $state(0);
  const choices = ['random', 'greedy', 'heuristic', 'search', 'checkpoint'];

  async function api<T>(url: string, body?: unknown): Promise<T> {
    const response = await recordFetch(url, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { 'content-type': 'application/json', 'x-etude-participant-token': token },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(typeof payload.detail === 'string' ? payload.detail.replaceAll('_', ' ') : `Request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }

  onMount(() => {
    ready = true;
    void fetch('/api/opponent').then(r => r.json()).then(data => {
      trained = data.available;
      if (data.opponent) trainedName = `${data.opponent.name} · ${data.opponent.sha256.slice(0, 10)}`;
    }).catch(() => {});
    const selected = new URLSearchParams(location.search).get('trace');
    if (selected) {
      id = selected;
      token = localStorage.getItem(`etude.watch.${id}`) ?? '';
      busy = true;
      void inspect(0).catch(fail).finally(() => busy = false);
    }
  });

  function fail(reason: unknown): void {
    error = reason instanceof Error ? reason.message : String(reason);
    playing = false;
  }

  async function inspect(index: number): Promise<void> {
    const next = await api<WatchDecision>(`/api/watch/${id}/decisions/${index}?reveal=${reveal}`);
    current = next;
    ordinal = index;
    count = next.count;
    finished = next.finished;
    revision = next.count;
  }

  async function start(): Promise<void> {
    if (busy) return;
    busy = true; playing = false; error = ''; reveal = true;
    try {
      const next = await api<{id: string; token: string}>('/api/watch', {
        players, hero_deck: hero, villain_deck: villain, seed, villain_sims: 16,
      });
      id = next.id; token = next.token; revision = 0; current = null; count = 0; finished = false;
      localStorage.setItem(`etude.watch.${id}`, token);
      rememberRecordCredential(token);
      history.replaceState(null, '', `/replay?watch=1&trace=${encodeURIComponent(id)}`);
    } catch (reason) { fail(reason); }
    finally { busy = false; }
    if (id && !error) await nextDecision();
  }

  async function nextDecision(): Promise<void> {
    if (busy || restored) return;
    busy = true; error = '';
    try {
      if (current && ordinal + 1 < count) await inspect(ordinal + 1);
      else if (!finished) {
        const next = await api<{revision: number; finished: boolean}>(`/api/watch/${id}/step`, {revision});
        revision = next.revision;
        finished = next.finished;
        await inspect(revision - 1);
      } else playing = false;
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }

  async function seek(index: number): Promise<void> {
    if (busy || restored) return;
    busy = true; playing = false;
    try { await inspect(index); }
    catch (reason) { fail(reason); }
    finally { busy = false; }
  }

  $effect(() => {
    if (!playing || busy || restored) return;
    const timer = setTimeout(() => void nextDecision(), 1000 / speed);
    return () => clearTimeout(timer);
  });

  async function anotherLine(): Promise<void> {
    if (!current) return;
    playing = false; busy = true; error = '';
    current = {...current, observation: current.decision.frame.projection, revealed: false};
    try {
      restored = await api<RestoredReplayDecision>(`/api/traces/${id}/decisions/${encodeURIComponent(current.decision.address)}`);
      phase = 'restored';
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }

  async function retry(offer: InteractionOffer): Promise<void> {
    if (!restored) return;
    busy = true; error = '';
    try {
      const result = await api<StudyRetryResponse>(`/api/traces/${id}/decisions/${encodeURIComponent(restored.address)}/retry`, {
        command: {...restored.command, command_id: `study.${crypto.randomUUID()}`, offer_id: offer.id},
      });
      attempt = result.attempt_id;
      branch = result.retry.projection;
      phase = 'retried';
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }

  async function returnToWatch(): Promise<void> {
    busy = true;
    try {
      if (attempt) await api(`/api/study-attempts/${attempt}/return`, {});
      restored = null; attempt = null; branch = null;
      await inspect(ordinal);
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }

  async function importGame(event: Event): Promise<void> {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    busy = true; playing = false; error = '';
    importedGames = [];
    try {
      const buffer = file.name.endsWith('.gz')
        ? await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text()
        : await file.text();
      let games: unknown[];
      try {
        const parsed: unknown = JSON.parse(buffer);
        games = Array.isArray(parsed) ? parsed : [parsed];
      } catch {
        games = buffer.trim().split('\n').map(line => JSON.parse(line));
      }
      importedGames = games;
      selectedGame = 0;
    } catch (reason) { fail(reason); }
    finally { busy = false; }
    if (importedGames.length === 1) await openImportedGame();
  }

  async function openImportedGame(): Promise<void> {
    busy = true; error = ''; reveal = true;
    try {
      const result = await api<{id: string; token: string}>('/api/watch/import', importedGames[selectedGame]);
      id = result.id; token = result.token;
      localStorage.setItem(`etude.watch.${id}`, token); rememberRecordCredential(token);
      history.replaceState(null, '', `/replay?watch=1&trace=${encodeURIComponent(id)}`);
      await inspect(0);
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }
</script>

<main class="mx-auto w-full max-w-[1400px] px-4 py-6">
  <div class="rounded-sm border border-line bg-panel p-6 shadow-sheet">
    <h1 class="type-title text-display">Watch two bots</h1>
    <p class="type-annotation mt-2 text-ink-2">Watch both hands, or switch to the acting bot's view. Each bot chooses using only its own information.</p>
    <fieldset disabled={!ready || busy || restored !== null} class="mt-6 flex flex-wrap items-end gap-4 border-b border-line pb-6">
      <legend class="sr-only">Bot matchup</legend>
      {#each [0, 1] as seat}
        <label class="flex flex-col gap-2 type-label">Seat {seat + 1}
          <select bind:value={players[seat]} class="rounded border border-line bg-field p-2">
            {#each choices as choice}<option value={choice} disabled={choice === 'checkpoint' && !trained}>{choice === 'checkpoint' ? trainedName : choice}</option>{/each}
          </select>
        </label>
      {/each}
      <DeckSelector {hero} {villain} heroLabel="Seat 1 deck" villainLabel="Seat 2 deck" onHeroChange={v => hero = v} onVillainChange={v => villain = v} />
      <label class="flex flex-col gap-2 type-label">Deal seed<input class="w-32 rounded border border-line bg-field p-2" bind:value={seed} inputmode="numeric" /></label>
      <button class="btn btn-primary" onclick={() => void start()}>Start watching</button>
      <label class="flex flex-col gap-2 type-caption">Open arena / Tournament game<input aria-label="Open arena game" type="file" accept=".json,.jsonl,.gz" onchange={e => void importGame(e)} /></label>
      {#if importedGames.length > 1}
        <label>Recorded game <select bind:value={selectedGame}>{#each importedGames as _, index}<option value={index}>Game {index + 1}</option>{/each}</select></label>
        <button class="btn btn-secondary" onclick={() => void openImportedGame()}>Open selected game</button>
      {/if}
    </fieldset>
    {#if error}<p role="alert" class="my-4 text-mountain-ink">{error}</p>{/if}
    {#if busy}<p role="status" class="my-4 type-caption">{current ? 'Advancing the authority…' : 'Preparing watch…'}</p>{/if}
    {#if current}
      <div class="flex flex-wrap items-center gap-3 py-4">
        <button class="btn btn-secondary" disabled={busy || ordinal === 0 || restored !== null} onclick={() => void seek(ordinal - 1)}>Previous decision</button>
        <button class="btn btn-primary" disabled={busy || restored !== null || (finished && ordinal + 1 === count)} onclick={() => { playing = false; void nextDecision(); }}>Step decision</button>
        <button class="btn btn-secondary" disabled={restored !== null || (finished && ordinal + 1 === count)} onclick={() => playing = !playing}>{playing ? 'Pause' : 'Play'}</button>
        <button class="btn btn-secondary" disabled={restored !== null || (finished && ordinal + 1 === count)} onclick={() => { speed = 16; playing = true; }}>Fast-forward</button>
        <label>Speed <select bind:value={speed}><option value={1}>1×</option><option value={4}>4×</option><option value={16}>16×</option></select></label>
        <span role="status">Decision {ordinal + 1} of {count} · Seat {current.decision.viewer + 1}{finished ? ` · Finished: ${current.winner === null ? 'draw' : `seat ${current.winner + 1} wins`}` : ''}</span>
        <label><input type="checkbox" bind:checked={reveal} disabled={busy || restored !== null} onchange={() => void seek(ordinal)} /> Show both hands</label>
      </div>
      <div class="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_350px]">
        <GameBoard observation={branch ?? restored?.frame.projection ?? current.observation} heroLabel={`Seat ${current.decision.viewer + 1} · ${current.players[current.decision.viewer]}`} villainLabel={`Seat ${2 - current.decision.viewer} · ${current.players[1 - current.decision.viewer]}`} />
        <aside class="border-l border-line pl-5">
          {#each current.identities as identity}
            <p class="type-caption text-ink-2">Seat {identity.seat + 1}: {identity.name}{identity.version ? ` · ${identity.version.slice(0, 12)}` : ''}</p>
          {/each}
          {#if current.revealed && !restored}<p class="type-rubric text-mountain-ink">Both hands · spectator view</p>{/if}
          {#if restored}
            <StudyPanel {phase} {restored} {busy} inspected errorMessage={error} onRetry={offer => void retry(offer)} onReturn={() => void returnToWatch()} />
          {:else}
            <DecisionAdvice mode="watch" frame={current.decision.frame} offers={current.decision.frame.offers} recorded={current.outputs} chosenOffer={current.decision.offer_id} scenarios={[]} selectedScenarioId="" evidence={null} deltas={null} status="ok" reason={null} advisorId="" computeId="" onSelectScenario={() => {}} />
            <button class="btn btn-secondary mt-4" disabled={busy || !token} onclick={() => void anotherLine()}>Try another line</button>
            {#if !token}<p class="type-caption mt-2">Study requires the original watch session.</p>{/if}
          {/if}
        </aside>
      </div>
    {:else}<p class="py-12 text-center type-annotation">Choose both seats, then start watching.</p>{/if}
  </div>
</main>
