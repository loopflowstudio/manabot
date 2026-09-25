<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { recordFetch } from '$lib/records';
  import type { GameSummary } from '$lib/types';

  let games = $state<GameSummary[]>([]);
  let loading = $state(false);
  let error = $state('');
  let nextCursor = $state<string | null>(null);
  let mounted = $state(false);
  let name = $state('');
  let profileStatus = $state('');
  let requestSequence = 0;
  const fields = ['q', 'scope', 'player', 'other', 'pairing', 'version', 'deck', 'status', 'result', 'after', 'before'] as const;
  const labels: Record<string, string> = { q: 'Search', scope: 'Scope', player: 'Player', other: 'Opponent', pairing: 'Players', version: 'Bot version', deck: 'Deck', status: 'Status', result: 'Result', after: 'From', before: 'Through' };
  const selected = $derived(fields.filter((key) => page.url.searchParams.has(key) && !(key === 'scope' && page.url.searchParams.get(key) === 'all')));
  const participantOptions = $derived([...new Map(games.flatMap((game) => game.players).map((p) => [p.player_id, p])).values()]);

  onMount(() => {
    mounted = true;
    void recordFetch('/api/player').then(async (response) => {
      if (!response.ok) throw new Error('Player name could not be loaded.');
      name = (await response.json()).name;
    }).catch((e) => { profileStatus = e.message; });
  });

  $effect(() => {
    const search = page.url.search;
    if (mounted) void load(search);
  });

  async function load(search: string): Promise<void> {
    const sequence = ++requestSequence;
    loading = true;
    error = '';
    try {
      const params = new URLSearchParams(search);
      if (!params.has('scope')) params.set('scope', 'all');
      const response = await recordFetch(`/api/traces?${params}`);
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Games could not be loaded.');
      }
      const rows = await response.json() as GameSummary[];
      if (sequence !== requestSequence) return;
      games = rows;
      nextCursor = response.headers.get('X-Etude-Next-Cursor');
    } catch (e) {
      if (sequence === requestSequence) { error = e instanceof Error ? e.message : 'Games could not be loaded.'; games = []; }
    } finally {
      if (sequence === requestSequence) loading = false;
    }
  }

  function filterUrl(key: string, value: string | null): string {
    const params = new URLSearchParams(page.url.search);
    params.delete('cursor');
    if (value) params.set(key, value); else params.delete(key);
    if (key === 'player' && !value) { params.delete('other'); params.delete('result'); }
    return `/games?${params}`;
  }

  function search(event: SubmitEvent): void {
    event.preventDefault();
    const data = new FormData(event.currentTarget as HTMLFormElement);
    const params = new URLSearchParams();
    for (const key of fields) {
      const value = String(data.get(key) ?? '').trim();
      if (value) params.set(key, value);
    }
    void goto(`/games?${params}`, { keepFocus: true, noScroll: true });
  }

  function resultLabel(game: GameSummary): string {
    if (game.status !== 'completed') return { active: 'In progress', stopped: 'Stopped', interrupted: 'Interrupted' }[game.status];
    if (game.winner === null) return 'Draw';
    return `${game.players.find((p) => p.seat === game.winner)?.name ?? `Seat ${game.winner + 1}`} won`;
  }

  async function saveName(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    profileStatus = '';
    try {
      const response = await recordFetch('/api/player', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name }) });
      if (!response.ok) throw new Error('Could not save your name. Try again.');
      name = (await response.json()).name;
      profileStatus = 'Saved for your next game. Earlier games keep their recorded names.';
    } catch (e) { profileStatus = e instanceof Error ? e.message : 'Name could not be saved.'; }
  }

  function replayUrl(id: string): string {
    return `/replay?trace=${encodeURIComponent(id)}&from=${encodeURIComponent(page.url.pathname + page.url.search)}`;
  }
</script>

<svelte:head><title>Games · Etude Fantasia</title></svelte:head>

<main class="mx-auto max-w-[1400px] px-4 py-8">
  <header class="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
    <div>
      <p class="type-rubric text-ink-2">The game archive</p>
      <h1 class="type-title text-4xl text-display">Games</h1>
      <p class="mt-2 text-ink-2">Every matchup has a story. Find a player, follow a bot, revisit a game.</p>
    </div>
    <a href="/" class="btn btn-primary">Play a game</a>
  </header>

  <details class="mb-5 rounded border border-line bg-panel p-3">
    <summary class="cursor-pointer">Your player name</summary>
    <form onsubmit={saveName} class="mt-3 flex flex-wrap items-end gap-3">
      <label class="flex flex-col gap-1">Display name<input class="rounded border border-line bg-field p-2" bind:value={name} maxlength="60" required /></label>
      <button class="btn btn-secondary" type="submit">Save name</button>
      <p role="status" class="w-full text-ink-2">{profileStatus}</p>
    </form>
    <p class="mt-2 text-sm text-ink-2">Your identity is kept in this browser. New games join shared history; completed replays can be viewed by everyone. Feedback stays private.</p>
  </details>

  {#key page.url.search}
    <form onsubmit={search} class="rounded border border-line bg-panel p-4">
      <div class="flex flex-wrap items-end gap-3">
        <label class="flex min-w-48 grow flex-col gap-1">Search games
          <input name="q" value={page.url.searchParams.get('q') ?? ''} placeholder="Player, bot, fingerprint, or game ID" maxlength="200" class="rounded border border-line bg-field p-2" />
        </label>
        <label class="flex flex-col gap-1">Show
          <select name="scope" value={page.url.searchParams.get('scope') ?? 'all'} class="rounded border border-line bg-field p-2"><option value="all">Everyone's games</option><option value="mine">My games</option></select>
        </label>
        <label class="flex flex-col gap-1">Players
          <select name="pairing" value={page.url.searchParams.get('pairing') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any pairing</option><option value="human-bot">Human vs Bot</option><option value="bot-bot">Bot vs Bot</option><option value="human-human">Human vs Human</option></select>
        </label>
        <label class="flex flex-col gap-1">Status
          <select name="status" value={page.url.searchParams.get('status') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any status</option><option value="completed">Completed</option><option value="active">In progress</option><option value="stopped">Stopped</option><option value="interrupted">Interrupted</option></select>
        </label>
        <button class="btn btn-primary" type="submit">Find games</button>
      </div>
      <details class="mt-4" open={fields.slice(2).some((key) => !['pairing', 'status'].includes(key) && page.url.searchParams.has(key))}>
        <summary class="cursor-pointer text-ink-2">Player, version, deck and date filters</summary>
        <div class="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {#each ['player', 'other'] as key}
            <label class="flex flex-col gap-1">{key === 'player' ? 'Player' : 'Other player'}
              <select name={key} value={page.url.searchParams.get(key) ?? ''} class="rounded border border-line bg-field p-2">
                <option value="">Any player</option>
                {#if page.url.searchParams.get(key) && !participantOptions.some((p) => p.player_id === page.url.searchParams.get(key))}<option value={page.url.searchParams.get(key)!}>Selected player</option>{/if}
                {#each participantOptions as p}<option value={p.player_id}>{p.name} · {p.kind} · {p.player_id.slice(-6)}</option>{/each}
              </select>
            </label>
          {/each}
          <label class="flex flex-col gap-1">Bot version
            <select name="version" value={page.url.searchParams.get('version') ?? ''} class="rounded border border-line bg-field p-2">
              <option value="">All versions</option>
              {#if page.url.searchParams.get('version') && !games.some((g) => g.players.some((p) => p.version === page.url.searchParams.get('version')))}<option value={page.url.searchParams.get('version')!}>Selected version</option>{/if}
              {#each [...new Map(games.flatMap((g) => g.players).filter((p) => p.version).map((p) => [p.version!, p])).values()] as p}<option value={p.version!}>{p.name} · {p.version!.slice(0, 10)}</option>{/each}
            </select>
          </label>
          <label class="flex flex-col gap-1">Deck<select name="deck" value={page.url.searchParams.get('deck') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any deck</option><option value="ur_lessons">UR Lessons</option><option value="gw_allies">GW Allies</option><option value="interactive">Interactive</option><option value="custom">Custom</option></select></label>
          <label class="flex flex-col gap-1">Result for selected player<select name="result" value={page.url.searchParams.get('result') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any result</option><option value="win">Won</option><option value="loss">Lost</option><option value="draw">Draw</option></select></label>
          <label class="flex flex-col gap-1">From (UTC)<input type="date" name="after" value={page.url.searchParams.get('after') ?? ''} class="rounded border border-line bg-field p-2" /></label>
          <label class="flex flex-col gap-1">Through (UTC)<input type="date" name="before" value={page.url.searchParams.get('before') ?? ''} class="rounded border border-line bg-field p-2" /></label>
        </div>
      </details>
    </form>
  {/key}

  <div class="my-4 flex flex-wrap items-center gap-2" aria-label="Selected filters">
    {#each selected as key}<a href={filterUrl(key, null)} class="max-w-full truncate rounded-full border border-line px-3 py-1 text-sm" aria-label={`Remove ${labels[key]} filter`}>{labels[key]}: {page.url.searchParams.get(key)} ×</a>{/each}
    {#if selected.length}<a href="/games" class="px-2 underline">Clear all</a>{/if}
    <button type="button" onclick={() => void load(page.url.search)} disabled={loading} class="btn btn-secondary ml-auto">Refresh</button>
  </div>

  {#if error}<p role="alert" class="rounded border border-mountain p-4">{error} <button class="underline" onclick={() => void load(page.url.search)}>Try again</button></p>{/if}
  <p role="status" class="mb-3 text-ink-2">{loading ? 'Loading games…' : `${games.length} ${games.length === 1 ? 'game' : 'games'} on this page`}</p>
  {#if !loading && !error && !games.length}<p class="rounded border border-line bg-panel p-8 text-center">No games match these filters. Try another player or clear the filters.</p>{/if}
  {#if games.length}
    <div class="overflow-x-auto rounded border border-line bg-panel" aria-busy={loading}>
      <table class="w-full text-left">
        <caption class="sr-only">Recorded games across players</caption>
        <thead class="border-b border-line text-ink-2"><tr><th class="p-4">Played</th><th class="p-4">Player 1</th><th class="p-4">Player 2</th><th class="p-4">Result / status</th><th class="p-4">Review</th></tr></thead>
        <tbody>
          {#each games as game (game.id)}
            <tr class="border-b border-line align-top" data-testid="game-history-row">
              <td class="p-4"><time datetime={game.timestamp ?? undefined}>{game.timestamp ? new Date(game.timestamp).toLocaleString() : 'Unknown date'}</time><div class="mt-1 text-xs text-ink-2">{game.origin.replaceAll('_', ' ')}</div></td>
              {#each [0, 1] as seat}
                {@const player = game.players.find((p) => p.seat === seat)}
                <td class="p-4">
                  {#if player}
                    <a href={filterUrl('player', player.player_id)} class="font-semibold underline decoration-line underline-offset-4">{player.name}</a>
                    <div class="mt-1 text-sm text-ink-2">{player.kind === 'human' ? 'Human' : 'Bot'} · {player.deck.replaceAll('_', ' ')}</div>
                    {#if player.version}<a href={filterUrl('version', player.version)} title={player.version} class="mt-1 block font-mono text-xs underline">{player.version.slice(0, 10)}</a>{/if}
                    {#if page.url.searchParams.has('player') && page.url.searchParams.get('player') !== player.player_id}<a class="mt-1 block text-xs underline" href={filterUrl('other', player.player_id)}>Filter this pairing</a>{/if}
                  {:else}<span class="text-ink-2">Unknown player</span>{/if}
                </td>
              {/each}
              <td class="p-4"><span class="font-semibold">{resultLabel(game)}</span>{#if game.status !== 'completed' && game.end_reason !== 'active'}<div class="text-sm text-ink-2">{game.end_reason?.replaceAll('_', ' ')}</div>{/if}<div class="mt-1 text-xs text-ink-2">{game.num_events} recorded steps</div></td>
              <td class="p-4">{#if game.replay_available}<a href={replayUrl(game.id)} class="underline" aria-label={`Replay game ${game.id}`}>Replay</a>{:else}<span class="text-sm text-ink-2">Replay private until completed</span>{/if}
                {#each game.feedback as note}<blockquote class="mt-2 max-w-64 break-words text-sm italic">{note.note}</blockquote>{/each}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
  <nav aria-label="Game history pages" class="mt-4 flex gap-4">
    {#if page.url.searchParams.has('cursor')}<a href={filterUrl('cursor', null)} class="btn btn-secondary">Newest games</a>{/if}
    {#if nextCursor && !loading && !error}<a href={filterUrl('cursor', nextCursor)} class="btn btn-secondary">Older games</a>{/if}
  </nav>
</main>
