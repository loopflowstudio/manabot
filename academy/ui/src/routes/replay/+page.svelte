<script lang="ts">
  /**
   * Replay, Watch and Review: a recorded game on the board, read-only, with a
   * transport bar where live play has its input and the decision strip under
   * it. At a recorded choice the board draws that choice's own frame, every
   * offer on its card and row with the recorded policy beside it; elsewhere
   * the recorded board. "Try another line" opens Study on the same board: the
   * historical offers become live against an isolated rules branch, and Return
   * restores the recorded decision. At an analysed decision "See its line"
   * draws the analysing policy's own line from there (its first choice, then
   * the policy and the game's opponent to the end of the next turn) on the
   * same board, and "Next key moment" walks the analysis's key moments.
   *
   * The address is the state: `trace`, `position`, `near` (0, 1 or chooser),
   * `view` (acting or spectator), plus `review` to start from the first
   * position and `watch` to start a bot game. The read cursor, Watch
   * production and Study authority are the stores'; this page only binds them
   * to the board.
   */
  import { currentGame, rememberCurrentGame } from '$lib/current-game.svelte';
  import { goto, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { tick, untrack } from 'svelte';
  import { recordFetch } from '$lib/records';
  import WorkStatus from '$lib/components/WorkStatus.svelte';
  import WatchSetup from '$lib/components/WatchSetup.svelte';
  import Feedback from '$lib/components/Feedback.svelte';
  import StudyPanel from '$lib/components/StudyPanel.svelte';
  import Board from '$lib/board/Board.svelte';
  import Transport from '$lib/board/Transport.svelte';
  import { moveList } from '$lib/board/moves';
  import { gameResult } from '$lib/board/result';
  import { shortName } from '$lib/opponent-name';
  import {
    analysedEvidence, analysedMarks, analysisShownMoves, bottomSeat, entryPosition, moveAt, moveMarks, offerEvidence, perspectiveOf, positionHeading, recordedFrame, replayLog, sinceRevision, stripMarks, type Perspective,
  } from '$lib/board/replay';
  import { analysisStatus, byEvent, keyMoments, scoreName, type AgentLine, type Analysis, type AnalysisChoices } from '$lib/analysis';
  import { agreement, duration, estimateArc, numberRows, playOrDraw, type GameNumbers } from '$lib/numbers';
  import { createReplayStore } from '$lib/replay.svelte';
  import type { RecordedPosition } from '$lib/recording';
  import { createStudyStore } from '$lib/study.svelte';
  import { createPresentationPlayer } from '$lib/presentation.svelte';
  import { mergePresentationLabels, presentationLabelsFromFrame, presentationLabelsFromObservation } from '$lib/presentation';
  import { checkpointIdentity, playerIdentity, recordedIdentity, type PlayerIdentity } from '$lib/player-identity';
  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { ExperienceFrame, InteractionOffer, TraceSummary } from '$lib/types';
  import type { StudySource, StudyRetryResponse, StudyRevealResponse, StudyPlanKind, StudyPlanPreviewResponse } from '$lib/study-runtime';

  const replayStore = createReplayStore();
  const studyStore = createStudyStore();
  const presentation = createPresentationPlayer();
  let summaries = $state<TraceSummary[]>([]);
  let listError = $state('');
  let studyButton = $state<HTMLButtonElement | null>(null);
  let notes = $state<Record<string, string>>({});
  let perspective = $state<Perspective>(0);
  let stripOpen = $state(true);
  let loaded = '';
  let lastHistoryKey = '';

  const current = $derived(replayStore.current);
  const index = $derived(replayStore.index);
  const positions = $derived(index?.positions ?? []);
  const selectedKey = $derived(positions[replayStore.selected]?.key ?? current?.key ?? '');
  const retrySearch = $derived(!!index?.mirror_retry && (index.mirror_retry.code.startsWith('search_') || index.mirror_retry.code.startsWith('mirror_')));
  const atLiveEdge = $derived(replayStore.selected === positions.length - 1 && current?.key === selectedKey);
  const inStudy = $derived(studyStore.phase !== 'score');
  // The analysing policy's line from one decision, drawn in place of the recording.
  let line = $state.raw<{ line: AgentLine; base: ExperienceFrame; from: number } | null>(null);
  let lineAt = $state(0);
  let lineBusy = $state(false);
  const locked = $derived(inStudy || studyStore.busy || !!line || lineBusy);

  // ———————————————— Seats and names

  // "You" in the move list is the person in a Play game, else seat 1.
  const heroSeat = $derived<0 | 1>(index?.players.find(player => player.kind === 'human')?.seat === 1 ? 1 : 0);
  const bottom = $derived(bottomSeat(perspective, positions, replayStore.selected));
  /**
   * A seat's recorded identity; a record that predates recorded players falls back to
   * its configured opponent. Seat 2 carries the checkpoint's producer and provenance.
   */
  function identity(seat: number): PlayerIdentity {
    const opponent = seat === 1 ? index?.opponent : null;
    const recorded = index?.players.find(player => player.seat === seat);
    const base = recorded ? recordedIdentity(recorded)
      : opponent ? checkpointIdentity(opponent) : playerIdentity({ name: `Seat ${seat + 1}` });
    return opponent ? { ...base, producer: opponent.producer, provenance: index?.opponent_checkpoint ?? undefined } : base;
  }
  // The person's own Play game names them "You"; two players of one name are told apart by seat.
  const nameOf = (seat: number): string => {
    const player = index?.players.find(candidate => candidate.seat === seat);
    if (player?.kind === 'human' && index?.can_feedback) return 'You';
    const name = identity(seat).compact;
    return name === identity(1 - seat).compact ? `${name} · seat ${seat + 1}` : name;
  };
  // Sentences use the short name ("Heuristic cast…"); the plate keeps its kind.
  const saidOf = (seat: number): string => {
    const name = nameOf(seat);
    if (name === 'You') return name;
    const said = shortName(name);
    return said === shortName(nameOf(1 - seat)) ? `${said} (seat ${seat + 1})` : said;
  };
  const deckOf = (seat: number): string | null => {
    const player = index?.players.find(candidate => candidate.seat === seat);
    return player && player.deck !== 'closed' ? player.deck_name ?? player.deck : null;
  };

  // ———————————————— What the board draws

  const recorded = $derived(current && index ? recordedFrame(current, index, bottom) : null);
  // Study draws the restored decision live, then the branch it played, on the same board.
  const studyFrame = $derived.by<ExperienceFrame | null>(() => {
    const restored = studyStore.restored;
    if (!inStudy || !restored) return null;
    const branch = studyStore.preview?.projection ?? studyStore.branchObservation;
    return branch ? { ...restored.frame, projection: branch, offers: [], prompt: null } : restored.frame;
  });
  const lineFrame = $derived.by<ExperienceFrame | null>(() => {
    const step = line?.line.steps[lineAt];
    return line && step ? { ...line.base, projection: step.projection, offers: [], prompt: null } : null;
  });
  const frame = $derived(studyFrame ?? lineFrame ?? recorded);
  const studying = $derived(studyFrame !== null);
  const branched = $derived(studying || lineFrame !== null);
  const viewerSeat = $derived(branched ? (frame!.projection.agent.player_index === 1 ? 1 : 0) : bottom);
  // ———————————————— Analysis: a policy's reading of your decisions

  let analyses = $state.raw<Analysis[] | null>(null);
  let analysisChoices = $state.raw<AnalysisChoices | null>(null);
  let analysisChoice = $state<string | null>(null);
  let analyseWith = $state('');
  let analysisError = $state('');
  let analysisPoll: ReturnType<typeof setTimeout> | undefined;
  const analysis = $derived(analyses?.find(item => item.id === analysisChoice) ?? analyses?.at(-1) ?? null);
  const analysed = $derived(byEvent(analysis));
  const analysisName = $derived(analysis ? shortName(analysis.policy.name) : '');
  const analysisScale = $derived.by(() => {
    const name = scoreName(analysis?.decisions[0]?.semantic ?? null);
    if (!analysis || !name) return null;
    return `${analysisName}'s ${name.toLowerCase()} from your seat · ${analysis.decisions.length} of ${analysis.total} decisions`;
  });
  const analysedHere = $derived(current?.kind === 'choice' && current.event !== undefined ? analysed.get(current.event) ?? null : null);
  const evidence = $derived(!branched && current?.decision && recorded
    ? analysedHere ? analysedEvidence(recorded, analysedHere, current.decision.offer_id, analysisName) : offerEvidence(recorded, current.outputs, current.decision.offer_id)
    : null);
  const canRetry = $derived(studyStore.phase === 'restored' && !studyStore.busy);

  // ———————————————— The game in numbers

  let numbers = $state.raw<GameNumbers | null>(null);
  const numbersRows = $derived(numbers ? numberRows(numbers, heroSeat) : []);
  const agreed = $derived(analysis ? agreement(analysis) : null);
  const arc = $derived(analysis ? estimateArc(analysis) : null);
  const played = $derived(!!index?.players.some(player => player.kind === 'human') && !index?.watch);

  async function loadNumbers(id: string): Promise<void> {
    numbers = null;
    if (!played || replayStore.index?.status === 'waiting') return;
    try {
      const read = await replayStore.api<{ numbers: GameNumbers | null }>(`/api/traces/${encodeURIComponent(id)}/numbers`);
      if (replayStore.index?.id === id) numbers = read.numbers;
    } catch { /* Review stands without its numbers. */ }
  }

  const entries = $derived(replayLog(positions, heroSeat, saidOf));
  const moves = $derived(moveList(entries, { you: saidOf(heroSeat), opponent: saidOf(1 - heroSeat) }, new Map(), []));
  const currentMove = $derived(moveAt(entries, positions, replayStore.selected));
  const marks = $derived(analysis ? analysedMarks(stripMarks(positions, heroSeat), positions, analysed) : stripMarks(positions, heroSeat));
  const marksByMove = $derived(moveMarks(entries, positions, analysed));
  const shownMoves = $derived(analysisShownMoves(entries, positions));
  const since = $derived(studying ? studyStore.restored?.revision ?? null : line ? null : sinceRevision(positions, replayStore.selected));

  // ———————————————— Key moments and its line

  const moments = $derived(analysis ? keyMoments(analysis) : []);
  const positionOf = (event: number): number => positions.findIndex(position => position.kind === 'choice' && position.event === event);
  const momentHere = $derived(current?.kind === 'choice' ? moments.find(moment => moment.event === current.event) ?? null : null);
  const nextMoment = $derived(moments.map(moment => positionOf(moment.event)).find(at => at > replayStore.selected) ?? null);
  const lineEnd = $derived.by(() => {
    if (!line) return '';
    const { end, winner } = line.line;
    if (end === 'turn') return 'to the end of the next turn';
    if (end === 'limit') return 'until its length limit';
    return winner === null ? 'to a draw' : winner === heroSeat ? 'to a win for your seat' : `to a win for ${saidOf(1 - heroSeat)}`;
  });

  // Play on from one of your decisions of a finished game you played: a new
  // Play game whose moves before it are this game's (`played_from`).
  const playOnHref = $derived(index?.can_feedback && !index.watch && index.status !== 'waiting'
    && current?.kind === 'choice' && current.event !== undefined && current.viewer === heroSeat
    ? `/?played_from=${encodeURIComponent(index.id)}&event=${current.event}` : null);

  async function seeLine(): Promise<void> {
    const id = index?.id;
    if (!id || !analysis || !analysedHere || !recorded) return;
    const base = recorded;
    const from = replayStore.selected;
    replayStore.pause(); lineBusy = true; analysisError = '';
    try {
      const read = await replayStore.api<AgentLine>(`/api/traces/${encodeURIComponent(id)}/analyses/${encodeURIComponent(analysis.id)}/line`, { event: analysedHere.event });
      if (replayStore.index?.id !== id || replayStore.selected !== from) return;
      line = { line: read, base, from }; lineAt = 0;
    } catch (reason) { analysisError = reason instanceof Error ? reason.message : String(reason); }
    finally { lineBusy = false; }
  }
  async function returnFromLine(): Promise<void> {
    line = null; lineAt = 0;
    await tick();
    document.querySelector<HTMLButtonElement>('[data-testid="see-line"]')?.focus();
  }

  const ending = $derived(frame && !branched && frame.projection.game_over && index
    ? gameResult(frame, { you: saidOf(viewerSeat), opponent: saidOf(1 - viewerSeat) }, index.winner)
    : null);

  const status = $derived(!index ? ''
    : replayStore.busy ? 'Reading saved recording…'
      : index.record_stop?.status === 'diverged' ? `Stopped at decision ${index.record_stop.index} · saved recording only`
        : index.status === 'finished' ? `Finished · ${index.winner === null ? 'Draw' : `${saidOf(index.winner)} wins`}`
          : index.status === 'waiting' ? index.watch ? atLiveEdge ? 'Waiting at the live edge' : 'Live Watch · reading saved recording' : 'Live game · saved recording only'
            : 'Interrupted · saved recording only');
  const sourceLabel = $derived(index?.source_kind === 'play' ? 'Recorded Play game' : index?.source_kind === 'training' ? 'Training game'
    : index?.source_kind ? 'Evaluation game' : index?.watch && !index.record_stop ? 'Watch' : 'Recorded game');
  const perspectiveText = $derived(!current ? ''
    : current.kind === 'choice' && current.decision
      ? `${current.revealed ? 'Both hands · spectator view' : `Acting view · seat ${current.viewer + 1}`} · chooser seat ${current.decision.viewer + 1}`
      : `Recorded seat-${current.viewer + 1} view · private post-action views unavailable`);

  // ———————————————— Opening a recording

  $effect(() => {
    const id = page.url.searchParams.get('trace');
    if (!id || id === loaded) return;
    loaded = id;
    untrack(() => { void open(id); });
  });
  $effect(() => { page.url.searchParams.has('watch'); untrack(() => { void loadList(); }); });

  async function loadList(): Promise<void> {
    try {
      const response = await recordFetch('/api/traces');
      if (!response.ok) throw new Error('Games could not be loaded.');
      summaries = await response.json(); listError = '';
    } catch (reason) { listError = reason instanceof Error ? reason.message : String(reason); }
  }

  async function open(id: string): Promise<void> {
    studyStore.resetSelection(); lastHistoryKey = '';
    let saved: { key?: string; near?: number | 'chooser'; reveal?: boolean } = {};
    try { saved = JSON.parse(sessionStorage.getItem(`etude.replay.${id}`) ?? '{}'); } catch { /* Invalid preferences select no private data. */ }
    const params = page.url.searchParams;
    // Review starts the game from its first position, whatever was read last.
    const review = params.has('review');
    replayStore.reveal = (params.get('view') ?? (saved.reveal ? 'spectator' : 'acting')) === 'spectator';
    try { notes = JSON.parse(localStorage.getItem(`etude.score.notes.${id}`) ?? '{}'); } catch { notes = {}; }
    await replayStore.open(id, review ? undefined : params.get('position') ?? saved.key);
    if (review && replayStore.index) await replayStore.seek(0);
    const near = params.get('near') ?? (saved.near === undefined ? null : String(saved.near));
    perspective = perspectiveOf(near, replayStore.index ?? { players: [] });
    if (!currentGame.href || replayStore.index?.can_produce) rememberCurrentGame(`/replay?trace=${encodeURIComponent(id)}`);
    await Promise.all([loadAnalyses(id, true), loadNumbers(id)]);
  }

  /**
   * Reads this game's analyses, following one that is running. Opening reads
   * who may analyse it once (it re-executes the game); following reads only the
   * saved analyses. Your own game, once nobody is playing it, opens its default
   * analysis if it has none.
   */
  async function loadAnalyses(id: string, opening = false): Promise<void> {
    clearTimeout(analysisPoll);
    const index = replayStore.index;
    if (!index || index.id !== id || index.watch) { analyses = null; analysisChoices = null; return; }
    const path = `/api/traces/${encodeURIComponent(id)}/analyses`;
    try {
      const choices = opening ? await replayStore.api<AnalysisChoices>(`${path}/policies`) : analysisChoices;
      let list = (await replayStore.api<{ analyses: Analysis[] }>(path)).analyses;
      if (opening && !list.length && choices?.default && index.can_feedback && index.status !== 'waiting') {
        await replayStore.api(path, { policy: choices.default });
        list = (await replayStore.api<{ analyses: Analysis[] }>(path)).analyses;
      }
      if (replayStore.index?.id !== id) return;
      analysisChoices = choices; analyses = list; analysisError = '';
      if (!analyseWith) analyseWith = choices?.default ?? choices?.policies[0]?.sha256 ?? '';
      if (list.some(item => item.status === 'running')) analysisPoll = setTimeout(() => void loadAnalyses(id), 1000);
    } catch (reason) { analysisError = reason instanceof Error ? reason.message : String(reason); }
  }

  async function analyse(policy: string): Promise<void> {
    const id = index?.id;
    if (!id || !policy) return;
    try {
      const started = await replayStore.api<{ id: string }>(`/api/traces/${encodeURIComponent(id)}/analyses`, { policy });
      analysisChoice = started.id;
      await loadAnalyses(id);
    } catch (reason) { analysisError = reason instanceof Error ? reason.message : String(reason); }
  }
  $effect(() => () => clearTimeout(analysisPoll));

  // The presentation track up to this position: the acting viewer's recorded history.
  $effect(() => {
    if (!current || !index || inStudy) return;
    untrack(() => {
      if (current.key === lastHistoryKey) return;
      lastHistoryKey = current.key;
      const batches = current.history.map((event, i) => ({ events: [event], labels: current.history_labels[i] }));
      presentation.loadPrefix(batches, `replay:${index.id}:viewer:${current.viewer}`, false);
      presentation.playing = false;
    });
  });

  // The address follows the position, the perspective and the view.
  $effect(() => {
    if (!current || !index || inStudy) return;
    const saved = { key: selectedKey, near: perspective, reveal: replayStore.reveal };
    untrack(() => {
      sessionStorage.setItem(`etude.replay.${index.id}`, JSON.stringify(saved));
      const url = new URL(location.href);
      url.searchParams.set('trace', index.id); url.searchParams.set('position', saved.key);
      url.searchParams.set('near', String(saved.near)); url.searchParams.set('view', saved.reveal ? 'spectator' : 'acting');
      url.searchParams.delete('review');
      replaceState(url, {});
    });
  });

  // ———————————————— Transport

  // Playback advances one position per beat; the board's motion draws each change.
  $effect(() => {
    if (!replayStore.playing || replayStore.busy || (replayStore.producing && atLiveEdge) || inStudy) return;
    const timer = setTimeout(() => { void replayStore.next(true); }, 1000 / replayStore.speed);
    return () => clearTimeout(timer);
  });
  $effect(() => () => replayStore.cancelReads());

  function seek(selected: number): void {
    if (inStudy || !positions.length) return;
    void replayStore.seek(Math.max(0, Math.min(positions.length - 1, selected)));
  }
  function next(): void {
    replayStore.pause();
    if (atLiveEdge && index?.mirror_retry) void replayStore.produce(true);
    else void replayStore.next();
  }
  function decision(direction: -1 | 1): void {
    const at = replayStore.selected;
    const target = direction < 0
      ? positions.findLastIndex((position, i) => i < at && position.kind === 'choice')
      : positions.findIndex((position, i) => i > at && position.kind === 'choice');
    if (target >= 0) seek(target);
  }
  function togglePlaying(): void {
    if (replayStore.playing) replayStore.pause();
    else replayStore.playing = true;
  }
  function jump(id: string): void {
    const key = entryPosition(id);
    const at = positions.findIndex(position => position.key === key);
    if (at >= 0) seek(at);
  }

  function onKeydown(event: KeyboardEvent): void {
    if (line && !event.metaKey && !event.ctrlKey && !event.altKey) {
      const last = line.line.steps.length - 1;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); lineAt = Math.max(0, Math.min(last, lineAt + (event.key === 'ArrowLeft' ? -1 : 1))); }
      else if (event.key === 'Escape') { event.preventDefault(); void returnFromLine(); }
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey || locked || !index) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
    const control = target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement || (target instanceof HTMLElement && target.tagName === 'SUMMARY');
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const direction = event.key === 'ArrowLeft' ? -1 : 1;
      if (event.shiftKey) decision(direction);
      else if (direction < 0) seek(replayStore.selected - 1);
      else next();
    } else if (event.key === 'Home') { event.preventDefault(); seek(0); }
    else if (event.key === 'End') { event.preventDefault(); seek(positions.length - 1); }
    else if (event.key === ' ' && !control) { event.preventDefault(); togglePlaying(); }
  }

  function setPerspective(value: string): void {
    perspective = value === 'chooser' ? 'chooser' : value === '1' ? 1 : 0;
  }
  function saveNote(key: string, value: string): void {
    notes[key] = value;
    if (index) localStorage.setItem(`etude.score.notes.${index.id}`, JSON.stringify(notes));
  }

  // ———————————————— Study: try another line on the same board

  async function responseError(response: Response, fallback: string): Promise<string> {
    try {
      const payload = (await response.json()) as { detail?: string };
      const messages: Record<string, string> = {
        study_branch_unavailable: 'Retry is unavailable because this recording no longer has its retained rules root.',
        study_command_not_structured: 'That historical offer is not yet available through the exact Retry authority.',
        study_command_identity_mismatch: 'The Retry offer no longer matches this recorded decision.',
        study_attempt_active: 'Return from the current Study attempt before starting another.',
        study_evidence_unavailable: 'Study evidence is unavailable for this recording.',
        study_evidence_not_revealed: 'Reveal the plans before previewing them.',
        study_attempt_not_found: 'This Study attempt expired. Return to the score and open the decision again.',
      };
      return payload.detail ? (messages[payload.detail] ?? payload.detail.replaceAll('_', ' ')) : fallback;
    } catch {
      return fallback;
    }
  }

  async function openStudyDecision(): Promise<void> {
    const decisionRow = replayStore.current?.decision;
    const traceId = replayStore.index?.id;
    if (!decisionRow || !traceId) return;
    replayStore.cancelReads();
    studyStore.setBusy(true); studyStore.setError(null);
    try {
      const selected = await replayStore.api<RecordedPosition & { study_source: StudySource }>(`/api/traces/${traceId}/positions/${encodeURIComponent(decisionRow.address)}?study=true`);
      const restored = await replayStore.api<RestoredReplayDecision>(`/api/traces/${traceId}/decisions/${encodeURIComponent(decisionRow.address)}`);
      if (replayStore.current?.key !== decisionRow.address) throw new Error('The selected choice changed while opening Study.');
      studyStore.restore(selected.study_source, decisionRow, restored);
      // Study is the acting view; a spectator reveal never carries into it or back.
      replayStore.reveal = false;
      if (replayStore.current) replayStore.current = { ...replayStore.current, observation: decisionRow.frame.projection, revealed: false };
    } catch (reason) { studyStore.setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { studyStore.setBusy(false); }
  }

  async function retryStudyOffer(offer: InteractionOffer): Promise<void> {
    const traceId = replayStore.index?.id;
    const restored = studyStore.restored;
    if (!traceId || !restored || !restored.frame.prompt || !canRetry) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const command = {
        command_id: `study.${crypto.randomUUID()}`,
        match_id: restored.frame.match_id,
        expected_revision: restored.revision,
        prompt_id: restored.frame.prompt.id,
        offer_id: offer.id,
        answers: [],
      };
      const response = await replayStore.request(`/api/traces/${traceId}/decisions/${encodeURIComponent(restored.address)}/retry`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ command }),
      });
      if (!response.ok) throw new Error(await responseError(response, 'Retry failed.'));
      const payload = (await response.json()) as StudyRetryResponse;
      studyStore.acceptRetry(payload);
      presentation.loadEvents(payload.retry.presentation,
        mergePresentationLabels(presentationLabelsFromFrame(restored.frame), presentationLabelsFromObservation(payload.retry.projection)),
        `replay:${traceId}:study:${payload.attempt_id}:retry`);
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Retry failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function revealStudy(): Promise<void> {
    if (!studyStore.attemptId) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const response = await replayStore.request(`/api/study-attempts/${studyStore.attemptId}/reveal`, { method: 'POST' });
      if (!response.ok) throw new Error(await responseError(response, 'Reveal failed.'));
      studyStore.reveal((await response.json()) as StudyRevealResponse);
      await previewStudyPlan('played');
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Reveal failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function previewStudyPlan(plan: StudyPlanKind): Promise<void> {
    if (!studyStore.attemptId) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const response = await replayStore.request(`/api/study-attempts/${studyStore.attemptId}/preview`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ plan }),
      });
      if (!response.ok) throw new Error(await responseError(response, 'Plan preview failed.'));
      const payload = (await response.json()) as StudyPlanPreviewResponse;
      studyStore.acceptPreview(payload);
      const restored = studyStore.restored;
      presentation.loadEvents(payload.presentation,
        restored ? mergePresentationLabels(presentationLabelsFromFrame(restored.frame), presentationLabelsFromObservation(payload.projection))
          : presentationLabelsFromObservation(payload.projection),
        `replay:${replayStore.index?.id}:study:${studyStore.attemptId}:${plan}`);
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Plan preview failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function returnToScore(): Promise<void> {
    studyStore.setBusy(true);
    const attemptId = studyStore.attemptId;
    let restoreFocus = false;
    try {
      let returned = studyStore.restored;
      if (attemptId) {
        const response = await replayStore.request(`/api/study-attempts/${attemptId}/return`, { method: 'POST' });
        if (!response.ok) throw new Error(await responseError(response, 'Return failed.'));
        returned = (await response.json()) as RestoredReplayDecision;
        const expected = studyStore.restored;
        if (!expected || JSON.stringify(returned) !== JSON.stringify(expected)) throw new Error('Returned decision differs from canonical replay.');
      }
      if (!returned) throw new Error('The canonical Study decision is unavailable.');
      studyStore.returnToScore(returned);
      lastHistoryKey = '';
      restoreFocus = true;
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Return failed.');
    } finally {
      studyStore.setBusy(false);
      if (restoreFocus) {
        await tick();
        studyButton?.focus();
      }
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#snippet notices()}
  {#if listError}<p role="alert" class="notice error">{listError}</p>{/if}
  {#if index?.mirror_retry && !replayStore.error}<p role="alert" class="notice">{index.mirror_retry.message}</p>{/if}
  {#if replayStore.error}<p role="alert" class="notice error">{replayStore.error}</p>{/if}
  {#if index?.record_stop?.reason}<p class="type-caption text-ink-2">{index.record_stop.reason}</p>{/if}
  {#if index?.played_from}
    <p class="type-caption text-ink-2" data-testid="played-from">Played on from <a href={`/replay?trace=${encodeURIComponent(index.played_from.trace_id)}&review=1`}>an earlier game</a>{index.played_from.turn === null ? '' : ` at turn ${index.played_from.turn}`}: the moves before it are that game's.</p>
  {/if}
  {#if analysisError}<p role="alert" class="notice error">{analysisError}</p>{/if}
  {#if analysis && analysisStatus(analysis)}<p class="type-caption text-ink-2" role="status" data-testid="analysis-status">{analysisStatus(analysis)}</p>{/if}
  {#if analyses && analysisChoices && !analysis && !index?.watch}
    <p class="type-caption text-ink-3" data-testid="analysis-absent">{analysisChoices.unavailable ?? (analysisChoices.policies.length ? 'No analysis yet · Analyse with… in the Menu' : 'No analysis: no registered policy can read this game. Install a policy provider (managym.bots) to analyse it.')}</p>
  {/if}
  {#if replayStore.producing && index}
    <section aria-label="Live frontier work" class="type-caption text-ink-2">
      <p>{atLiveEdge ? 'Producing the next decision.' : 'Reading a saved position; bot work belongs to the frontier.'}</p>
      {#if replayStore.workDisconnected}<p role="status">Connection unavailable; checking the same live frontier. No retry has been sent.</p>{/if}
      <WorkStatus work={replayStore.work} connected={!replayStore.workDisconnected} />
    </section>
  {/if}
  {#if line}
    <section class="line" aria-label="Its line" data-testid="agent-line">
      <p class="type-label text-ink-2">{analysisName}'s line from here</p>
      <p class="type-caption text-ink-2">One play-out, {lineEnd}: {analysisName}'s first choice in your seat, {saidOf(1 - heroSeat)} in theirs. The game as played is unchanged.</p>
      <ol class="line-moves type-caption">
        {#each line.line.steps as step, at (at)}
          <li><button class:current={at === lineAt} aria-current={at === lineAt ? 'step' : undefined} data-testid="line-move" onclick={() => (lineAt = at)}>
            <span class="text-ink-3">{step.actor === 'agent' ? analysisName : saidOf(1 - heroSeat)}</span> {step.description}</button></li>
        {/each}
      </ol>
      <p class="line-controls">
        <button class="btn btn-ghost btn-sm" disabled={lineAt === 0} onclick={() => (lineAt -= 1)}>Previous</button>
        <button class="btn btn-ghost btn-sm" disabled={lineAt === line.line.steps.length - 1} onclick={() => (lineAt += 1)}>Next</button>
        <button class="btn btn-secondary btn-sm" data-testid="line-return" onclick={() => void returnFromLine()}>Return to the game</button>
      </p>
    </section>
  {:else if studying && studyStore.restored}
    <StudyPanel phase={studyStore.phase} restored={studyStore.restored} plans={studyStore.plans} selectedPlan={studyStore.selectedPlan} busy={studyStore.busy}
      errorMessage={studyStore.errorMessage} onReveal={() => void revealStudy()} onReturn={() => void returnToScore()} onPreview={plan => void previewStudyPlan(plan)} />
  {:else if current}
    {#if studyStore.errorMessage}<p role="alert" class="notice error">{studyStore.errorMessage}</p>{/if}
    {#if current.description}<p class="type-caption text-ink-2" data-testid="position-description">After · {current.description}</p>{/if}
    {#if momentHere}<p class="type-caption text-ink-2" data-testid="key-moment">Key moment · {momentHere.reason}</p>{/if}
    {#if (analysedHere && index?.can_feedback) || playOnHref || (current.decision && index?.can_study)}
      <p class="analysis-actions">
        {#if analysedHere && index?.can_feedback && analysis}
          <button class="btn btn-secondary btn-sm" data-testid="see-line" disabled={lineBusy || replayStore.busy} onclick={() => void seeLine()}>{lineBusy ? 'Playing its line…' : 'See its line'}</button>
        {/if}
        {#if playOnHref}
          <a class="btn btn-secondary btn-sm" data-testid="play-on" href={playOnHref}>Play on from here</a>
        {/if}
        {#if current.decision && index?.can_study}
          <button class="btn btn-secondary btn-sm" data-testid="study-open" disabled={studyStore.busy || replayStore.busy || !!index.mirror_retry}
            onclick={() => void openStudyDecision()} bind:this={studyButton}>Try another line</button>
        {/if}
      </p>
    {/if}
    {#if !current.decision && current.kind === 'choice'}
      <p class="type-caption text-ink-3">Legal offers and Study not recorded.</p>
    {/if}
    {#if !current.consequences_recorded}<p class="type-caption text-ink-3">Consequences not recorded.</p>{/if}
    {#if current.views_stopped}<p class="type-caption text-ink-3" data-testid="views-stopped">{current.views_stopped}</p>{/if}
  {/if}
{/snippet}

{#snippet menu()}
  <label class="perspective type-caption">
    <span class="sr-only">Perspective</span>
    <select aria-label="Perspective" data-testid="perspective" value={String(perspective)} disabled={locked} onchange={event => setPerspective(event.currentTarget.value)}>
      <option value="chooser">Follow the chooser</option>
      <option value="0">Seat 1 at the bottom · {nameOf(0)}</option>
      <option value="1">Seat 2 at the bottom · {nameOf(1)}</option>
    </select>
  </label>
  {#if current?.can_reveal}
    <label class="type-caption reveal"><input type="checkbox" checked={current.revealed} disabled={locked || replayStore.busy}
      onchange={event => { replayStore.reveal = event.currentTarget.checked; void replayStore.seek(replayStore.selected); }} /> Show both hands</label>
  {/if}
  <a class="type-caption text-ink-2 hover:text-ink" href={page.url.searchParams.get('from')?.startsWith('/games?') ? page.url.searchParams.get('from')! : '/games'}>Games</a>
  <details class="game-menu">
    <summary class="btn btn-ghost btn-sm" data-testid="game-menu">Menu</summary>
    <div class="menu-panel" role="region" aria-label="Recording and tools">
      <p class="type-label text-ink-2" data-testid="recording-heading">{sourceLabel}</p>
      <p role="status" class="type-caption text-ink-2" data-testid="recording-status">{status}</p>
      <p class="type-caption text-ink-3" data-testid="replay-perspective">{perspectiveText}</p>
      <a class="btn btn-secondary btn-sm" href="/?setup=1">New game</a>
      {#if analyses && analysisChoices && !index?.watch}
        <section class="analysis-menu type-caption" aria-label="Analysis" data-testid="analysis-menu">
          <p class="type-label text-ink-2">Analysis</p>
          {#if analyses.length > 1}
            <label>Shown
              <select aria-label="Shown analysis" value={analysis?.id} onchange={event => (analysisChoice = event.currentTarget.value)}>
                {#each analyses as item, at (item.id)}<option value={item.id}>{at + 1}. {item.policy.name} · {item.status}</option>{/each}
              </select>
            </label>
          {:else if analysis}<p class="text-ink-2">By {analysis.policy.name}</p>{/if}
          {#if analysisChoices.unavailable}<p class="text-ink-3">{analysisChoices.unavailable}</p>
          {:else if analysisChoices.policies.length && index?.can_feedback}
            <div class="analyse-with">
              <select aria-label="Policy to analyse with" bind:value={analyseWith} disabled={analyses.some(item => item.status === 'running')}>
                {#each analysisChoices.policies as policy (policy.sha256)}<option value={policy.sha256}>{policy.name}</option>{/each}
              </select>
              <button class="btn btn-secondary btn-sm" data-testid="analyse-with" disabled={!analyseWith || analyses.some(item => item.status === 'running') || index?.status === 'waiting'}
                onclick={() => void analyse(analyseWith)}>{analysis?.status === 'stopped' && analysis.policy.id === analyseWith ? 'Analyse again' : 'Analyse with…'}</button>
            </div>
          {/if}
        </section>
      {/if}
      {#if played && index}
        <section class="numbers type-caption" aria-label="This game in numbers" data-testid="game-numbers">
          <p class="type-label text-ink-2">This game</p>
          <p class="text-ink-2">{[numbers ? `${numbers.turns} ${numbers.turns === 1 ? 'turn' : 'turns'}` : null, playOrDraw(heroSeat), duration(index.started_at, index.ended_at)].filter(Boolean).join(' · ')}</p>
          {#if numbersRows.length}
            <table>
              <thead><tr><th scope="col"><span class="sr-only">Count</span></th><th scope="col">{saidOf(heroSeat)}</th><th scope="col">{saidOf(1 - heroSeat)}</th></tr></thead>
              <tbody>{#each numbersRows as row (row.label)}<tr><th scope="row">{row.label}</th><td>{row.you}</td><td>{row.opponent}</td></tr>{/each}</tbody>
            </table>
            {#if numbers && !numbers.complete}<p class="text-ink-3">Counted to where the game stopped.</p>{/if}
            {#if numbers?.analysis_shown}<p class="text-ink-2" data-testid="analysis-shown-count">Chosen with analysis shown: {numbers.analysis_shown} {numbers.analysis_shown === 1 ? 'decision' : 'decisions'}</p>{/if}
          {:else if index.status !== 'waiting'}<p class="text-ink-3">Counts need the build that recorded this game.</p>{/if}
          {#if analysis && agreed}
            <p data-testid="agreement">Decisions with a choice: {agreed.of} · chose as {analysisName} would: {agreed.same} of {agreed.of}</p>
            {#if arc && analysisScale}<p class="text-ink-2" data-testid="estimate-arc">{analysisName}'s {scoreName(analysis.decisions[0]?.semantic ?? null)?.toLowerCase()} from your seat: {arc.start} at the start · {arc.peak} at its highest · {arc.end} at the end</p>{/if}
          {/if}
        </section>
      {/if}
      {#each [0, 1] as seat (seat)}
        {@const who = identity(seat)}
        {@const runtime = who.runtime ?? who.provenance?.policy}
        <details class="player-status type-caption">
          <summary>Seat {seat + 1} · {nameOf(seat)}</summary>
          <div class="identity" data-testid="player-identity">
            <p>{who.full}</p>
            {#if who.digest}<p>{runtime ? 'Policy ID' : 'SHA-256'}: {who.digest}</p>{/if}
            {#if who.producer}<p>Producer: {who.producer}</p>{/if}
            {#if runtime}
              <p>Runner: {runtime.runner}</p>
              {#each Object.entries(runtime.artifacts) as [role, digest] (role)}<p>{role}: {digest}</p>{/each}
              <p>Config: {JSON.stringify(runtime.config)}</p>
            {/if}
            {#each ['source', 'world', 'model', 'step'] as const as key (key)}
              {@const value = who.provenance?.[key]}
              {#if value != null}<p>{key}: {value}</p>{/if}
            {/each}
          </div>
        </details>
      {/each}
      <button class="btn btn-secondary btn-sm" disabled={locked} onclick={() => { if (replayStore.headUncertain) void replayStore.refreshHead(); else if (index) void replayStore.open(index.id, selectedKey); }}>Refresh saved head</button>
      {#if current}
        <label class="type-caption">Note for this position
          <textarea class="note" aria-label="Note for this position" value={notes[current.note_key] ?? ''} oninput={event => saveNote(current.note_key, event.currentTarget.value)}></textarea>
        </label>
      {/if}
      {#if index?.watch}
        <details><summary class="type-label cursor-pointer">Watch another matchup</summary>
          {#key index.id}<WatchSetup setup={index.private_setup} disabled={locked} onOpen={id => void goto(`/replay?trace=${encodeURIComponent(id)}`)} />{/key}
        </details>
      {/if}
      {#if index?.can_feedback}<Feedback attemptId={index.id} />{/if}
    </div>
  </details>
{/snippet}

{#snippet result()}
  {#if ending}
    <div data-testid="game-result-dialog" class="result">
      <p class="type-label text-ink-3">Game over · recorded</p>
      <h2 data-testid="game-result" class="type-display text-display">{ending.headline}</h2>
      <p class="text-ink-2">{ending.how}</p>
      <p class="type-caption text-ink-2">{ending.life}</p>
      <p class="type-caption"><button class="underline" onclick={() => seek(0)}>Review from the start</button> · <a class="underline" href="/">Play</a></p>
    </div>
  {/if}
{/snippet}

{#snippet bottomEdge()}
  {#if index}
    <Transport selected={replayStore.selected} total={positions.length} {marks} playing={replayStore.playing} speed={replayStore.speed}
      disabled={locked} producing={replayStore.producing} busy={replayStore.busy} canProduce={index.can_produce && !replayStore.headUncertain} {retrySearch}
      retryBot={!!index.mirror_retry && !retrySearch} line={!stripOpen} onPrevious={() => seek(replayStore.selected - 1)} onNext={next} onDecision={decision}
      onTogglePlaying={togglePlaying} onScrub={seek} onSpeedChange={speed => (replayStore.speed = speed)}
      onKeyMoment={!moments.length ? undefined : nextMoment === null ? null : () => seek(nextMoment!)} />
  {/if}
{/snippet}

<h1 class="sr-only">Replay</h1>
{#if index && frame}
  <Board
    {frame}
    live={studying && canRetry}
    status={replayStore.busy || atLiveEdge || studying ? status : ''}
    you={{ name: nameOf(viewerSeat), deck: deckOf(viewerSeat) }}
    opponent={{ name: nameOf(1 - viewerSeat), deck: deckOf(1 - viewerSeat), description: identity(1 - viewerSeat).description }}
    stops={frame.stops}
    log={[]}
    canPassTurn={false}
    {presentation}
    thinking={null}
    opponentChoosing={false}
    workStarted={null}
    interrupted={false}
    onact={offer => void retryStudyOffer(offer)}
    {since}
    recapTitle={studying ? 'This line' : line ? 'Its line' : current?.kind === 'consequences' ? 'This move' : 'Since the last decision'}
    heading={branched || !current ? null : positionHeading(current, saidOf, frame.projection.game_over, index?.winner ?? null)}
    {moves}
    onjump={locked ? undefined : jump}
    readOnly={!studying}
    {currentMove}
    {evidence}
    moveMarks={marksByMove}
    {shownMoves}
    bottom={bottomEdge}
    strip={index ? { marks, selected: replayStore.selected, open: stripOpen, ontoggle: () => (stripOpen = !stripOpen), onjump: locked ? undefined : seek, scale: analysisScale } : undefined}
    {notices}
    result={ending ? result : undefined}
    {menu}
  />
{:else}
  <main class="landing mx-auto w-full max-w-[900px] px-4 py-6">
    <header class="mb-4 flex flex-wrap items-baseline gap-3">
      <p class="type-title text-display" aria-hidden="true">Replay</p>
      <a class="underline type-caption" href="/games">Games</a>
      <a class="underline type-caption" href="/">Play</a>
    </header>
    {#if listError}<p role="alert">{listError}</p>{/if}
    {#if replayStore.error}<p role="alert">{replayStore.error}</p>{/if}
    {#if page.url.searchParams.has('watch')}
      <section class="mb-6 rounded border border-line bg-panel p-4" aria-label="Choose a bot matchup">
        <h3 class="type-label mb-2">Choose a bot matchup</h3>
        <WatchSetup disabled={false} onOpen={id => void goto(`/replay?trace=${encodeURIComponent(id)}`)} />
      </section>
    {/if}
    {#if replayStore.busy}<p role="status">Reading saved recording…</p>
    {:else if !page.url.searchParams.has('trace')}
      <section aria-label="Recent recordings">
        <h3 class="type-label mb-2">Recent recordings</h3>
        <ol class="recordings" data-testid="recording-list">
          {#each summaries.slice(0, 30) as summary (summary.id)}
            <li><a class="underline" data-testid="recording-link" href={`/replay?trace=${encodeURIComponent(summary.id)}`}>{summary.timestamp ? new Date(summary.timestamp).toLocaleString() : summary.id}</a></li>
          {:else}
            <li class="type-caption text-ink-2">No recordings yet. Finished games appear here and in Games.</li>
          {/each}
        </ol>
      </section>
    {/if}
  </main>
{/if}

<style>
  .notice { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg-field); font-size: 13px; }
  .notice.error { border-color: color-mix(in srgb, var(--error) 50%, transparent); background: var(--error-wash); color: var(--error-text); }
  .perspective select { border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 2px 4px; color: var(--text); max-width: 220px; }
  .reveal { display: inline-flex; gap: 4px; align-items: center; color: var(--text-secondary); white-space: nowrap; }
  .result { display: grid; gap: 6px; justify-items: start; }
  .result h2 { line-height: 1.1; }
  .game-menu { position: relative; }
  .game-menu > summary { list-style: none; }
  .game-menu > summary::-webkit-details-marker { display: none; }
  .menu-panel {
    position: absolute; right: 0; top: calc(100% + 8px); z-index: 320; width: 360px; max-height: calc(100dvh - 64px); overflow-y: auto;
    display: grid; gap: 10px; padding: 14px; border: 1px solid var(--border-strong); border-radius: 10px;
    background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .menu-panel > a.btn { justify-self: start; }
  .note { display: block; width: 100%; margin-top: 4px; min-height: 60px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 4px 6px; }
  .recordings { display: grid; gap: 4px; }
  .analysis-menu { display: grid; gap: 4px; }
  .analysis-menu select { border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 2px 4px; color: var(--text); max-width: 220px; }
  .analyse-with { display: flex; gap: 6px; align-items: center; }
  .numbers { display: grid; gap: 4px; }
  .analysis-actions, .line-controls { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .line { display: grid; gap: 6px; }
  .line-moves { display: grid; gap: 1px; max-height: 180px; overflow-y: auto; }
  .line-moves button { text-align: left; padding: 1px 6px; border-radius: 4px; width: 100%; }
  .line-moves button.current { background: var(--bg-field); color: var(--text); }
  .numbers table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .numbers th, .numbers td { padding: 1px 8px 1px 0; text-align: right; font-weight: normal; }
  .numbers th[scope='row'] { text-align: left; color: var(--text-secondary); }
  .player-status summary { cursor: pointer; }
  .identity { padding: 4px 0 0 12px; overflow-wrap: anywhere; color: var(--text-secondary); }
</style>
