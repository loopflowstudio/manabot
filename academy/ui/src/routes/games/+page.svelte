<script lang="ts">
  /**
   * History: every game in the traces directory, newest first. Each row opens
   * in Replay; an unfinished played game also resumes at `/?game=<id>`. The
   * address holds the search and filters.
   */
  import GameFileImport from '$lib/components/GameFileImport.svelte';
  import PlayerName from '$lib/components/PlayerName.svelte';
  import { recordedIdentity, shortHash } from '$lib/player-identity';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import type { GameSummary } from '$lib/types';
  import { duration } from '$lib/numbers';

  let games = $state<GameSummary[]>([]);
  let loading = $state(false);
  let error = $state('');
  let nextCursor = $state<string | null>(null);
  let mounted = $state(false);
  let requestSequence = 0;
  const fields = ['q', 'pairing', 'status', 'bot', 'deck', 'after', 'before'] as const;
  const labels: Record<string, string> = { q: 'Search', pairing: 'Players', status: 'Status', bot: 'Bot', deck: 'Deck', after: 'From', before: 'Through' };
  const selected = $derived(fields.filter(key => page.url.searchParams.has(key)));
  // The filters offer what this page of games shows; a selection from another page keeps its own option.
  const bots = $derived([...new Map(games.flatMap(game => game.players).filter(player => player.kind === 'bot').map(player => [player.player_id, player])).values()]);
  const decks = $derived([...new Set(games.flatMap(game => game.players).map(player => player.deck).filter(deck => deck && deck !== 'closed'))].sort());

  onMount(() => { mounted = true; });

  $effect(() => {
    const search = page.url.search;
    if (mounted) void load(search);
  });

  async function load(search: string): Promise<void> {
    const sequence = ++requestSequence;
    loading = true;
    error = '';
    try {
      const response = await fetch(`/api/traces${search}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
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

  function count(amount: number, singular: string, plural = `${singular}s`): string {
    return `${amount} ${amount === 1 ? singular : plural}`;
  }

  function filterUrl(key: string, value: string | null): string {
    const params = new URLSearchParams(page.url.search);
    params.delete('cursor');
    if (value) params.set(key, value); else params.delete(key);
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

  /** In progress covers every unfinished game, however an earlier build labelled it. */
  function resultLabel(game: GameSummary): string {
    if (game.status === 'active') return 'In progress';
    if (game.status !== 'completed') return 'Stopped';
    if (game.winner === null) return 'Draw';
    const winner = game.players.find(player => player.seat === game.winner);
    return `${winner ? recordedIdentity(winner).compact : `Seat ${game.winner + 1}`} won`;
  }

  const here = $derived(encodeURIComponent(page.url.pathname + page.url.search));
  const replayUrl = (id: string): string => `/replay?trace=${encodeURIComponent(id)}&from=${here}`;
</script>

<svelte:head><title>History · Academy</title></svelte:head>

<main class="mx-auto max-w-[1400px] px-4 py-8">
  <header class="mb-6 border-b border-line pb-5">
    <h1 class="type-title text-4xl text-display">History</h1>
  </header>

  <GameFileImport onOpen={id => void goto(replayUrl(id))} />

  {#key page.url.search}
    <form onsubmit={search} class="rounded border border-line bg-panel p-4">
      <div class="flex flex-wrap items-end gap-3">
        <label class="flex min-w-48 grow flex-col gap-1">Search games
          <input name="q" value={page.url.searchParams.get('q') ?? ''} placeholder="Bot, fingerprint, or game ID" maxlength="200" class="rounded border border-line bg-field p-2" />
        </label>
        <label class="flex flex-col gap-1">Players
          <select name="pairing" value={page.url.searchParams.get('pairing') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any pairing</option><option value="human-bot">Human vs Bot</option><option value="bot-bot">Bot vs Bot</option></select>
        </label>
        <label class="flex flex-col gap-1">Status
          <select name="status" value={page.url.searchParams.get('status') ?? ''} class="rounded border border-line bg-field p-2"><option value="">Any status</option><option value="active">In progress</option><option value="completed">Completed</option><option value="stopped">Stopped</option></select>
        </label>
        <button class="btn btn-primary" type="submit">Find games</button>
      </div>
      <details class="mt-4" open={(['bot', 'deck', 'after', 'before'] as const).some(key => page.url.searchParams.has(key))}>
        <summary class="cursor-pointer text-ink-2">Bot, deck and date filters</summary>
        <div class="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label class="flex flex-col gap-1">Bot
            <select name="bot" value={page.url.searchParams.get('bot') ?? ''} class="rounded border border-line bg-field p-2">
              <option value="">Any bot</option>
              {#if page.url.searchParams.get('bot') && !bots.some(bot => bot.player_id === page.url.searchParams.get('bot'))}<option value={page.url.searchParams.get('bot')!}>Selected bot</option>{/if}
              {#each bots as bot (bot.player_id)}<option value={bot.player_id}>{recordedIdentity(bot).compact}{bot.version ? ` · ${shortHash(bot.version)}` : ''}</option>{/each}
            </select>
          </label>
          <label class="flex flex-col gap-1">Deck
            <select name="deck" value={page.url.searchParams.get('deck') ?? ''} class="rounded border border-line bg-field p-2">
              <option value="">Any deck</option>
              {#if page.url.searchParams.get('deck') && !decks.includes(page.url.searchParams.get('deck')!)}<option value={page.url.searchParams.get('deck')!}>{page.url.searchParams.get('deck')!.replaceAll('_', ' ')}</option>{/if}
              {#each decks as deck (deck)}<option value={deck}>{deck.replaceAll('_', ' ')}</option>{/each}
            </select>
          </label>
          <label class="flex flex-col gap-1">From (UTC)<input type="date" name="after" value={page.url.searchParams.get('after') ?? ''} class="rounded border border-line bg-field p-2" /></label>
          <label class="flex flex-col gap-1">Through (UTC)<input type="date" name="before" value={page.url.searchParams.get('before') ?? ''} class="rounded border border-line bg-field p-2" /></label>
        </div>
      </details>
    </form>
  {/key}

  <div class="my-4 flex flex-wrap items-center gap-2" aria-label="Selected filters">
    {#each selected as key (key)}<a href={filterUrl(key, null)} class="max-w-full truncate rounded-full border border-line px-3 py-1 text-sm" aria-label={`Remove ${labels[key]} filter`}>{labels[key]}: {page.url.searchParams.get(key)} ×</a>{/each}
    {#if selected.length}<a href="/games" class="px-2 underline">Clear all</a>{/if}
    <button type="button" onclick={() => void load(page.url.search)} disabled={loading} class="btn btn-secondary ml-auto">Refresh</button>
  </div>

  {#if error}<p role="alert" class="rounded border border-mountain p-4">{error} <button class="underline" onclick={() => void load(page.url.search)}>Try again</button></p>{/if}
  <p role="status" class="mb-3 text-ink-2">{loading ? 'Loading games…' : `${count(games.length, 'game')} on this page`}</p>
  {#if !loading && !error && !games.length}
    <p class="rounded border border-line bg-panel p-8 text-center" data-testid="history-empty">{selected.length ? 'No games match these filters.' : 'No games yet. Play one, or open a recorded game file above.'}</p>
  {/if}
  {#if games.length}
    <div class="overflow-x-auto rounded border border-line bg-panel" aria-busy={loading}>
      <table class="w-full table-fixed text-left [overflow-wrap:anywhere]">
        <caption class="sr-only">Recorded games, newest first</caption>
        <colgroup><col style="width: 14%" /><col style="width: 28%" /><col style="width: 28%" /><col style="width: 18%" /><col style="width: 12%" /></colgroup>
        <thead class="border-b border-line text-ink-2"><tr><th class="p-4">Played</th><th class="p-4">Player 1 · on the play</th><th class="p-4">Player 2</th><th class="p-4">Result / status</th><th class="p-4">Open</th></tr></thead>
        <tbody>
          {#each games as game (game.id)}
            <tr class="border-b border-line align-top" data-testid="game-history-row">
              <td class="p-4"><time datetime={game.timestamp ?? undefined}>{game.timestamp ? new Date(game.timestamp).toLocaleString() : 'Unknown date'}</time><div class="mt-1 text-xs text-ink-2">{game.origin.replaceAll('_', ' ')}</div></td>
              {#each [0, 1] as seat (seat)}
                {@const player = game.players.find(candidate => candidate.seat === seat)}
                <td class="p-4">
                  {#if player}
                    {#if player.kind === 'bot'}
                      <a href={filterUrl('bot', player.player_id)} class="font-semibold underline decoration-line underline-offset-4"><PlayerName identity={recordedIdentity(player)} chip={false} /></a>
                    {:else}<span class="font-semibold"><PlayerName identity={recordedIdentity(player)} chip={false} /></span>{/if}
                    <div class="mt-1 text-sm text-ink-2">{player.kind === 'human' ? 'Human' : 'Bot'} · {player.deck.replaceAll('_', ' ')}</div>
                    {#if player.version}<span title={player.version} class="mt-1 block font-mono text-xs text-ink-2">{shortHash(player.version)}</span>{/if}
                  {:else}<span class="text-ink-2">Unknown player</span>{/if}
                </td>
              {/each}
              <td class="p-4"><span class="font-semibold" data-testid="game-history-status">{resultLabel(game)}</span>{#if game.status === 'stopped' && game.end_reason}<div class="text-sm text-ink-2">{game.end_reason.replaceAll('_', ' ')}</div>{/if}<div class="mt-1 text-xs text-ink-2" data-testid="game-history-numbers">{[game.numbers ? `${game.numbers.turns} ${game.numbers.turns === 1 ? 'turn' : 'turns'}` : null, duration(game.timestamp, game.ended_at), `${game.num_events} recorded steps`, game.numbers?.analysis_shown ? `analysis shown at ${count(game.numbers.analysis_shown, 'decision')}` : null, game.numbers?.played_from != null ? 'played on from another game' : null].filter(Boolean).join(' · ')}</div></td>
              <td class="p-4">
                {#if game.resumable}<a href={`/?game=${encodeURIComponent(game.id)}`} class="mr-3 font-semibold underline" data-testid="game-history-resume" aria-label={`Resume game ${game.id}`}>Resume</a>{/if}
                <a href={replayUrl(game.id)} class="underline" data-testid="game-history-replay" aria-label={`Replay game ${game.id}`}>Replay</a>
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
