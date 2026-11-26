<script lang="ts">
  /**
   * The game screen: the board in the centre, the rail beside it, the stop bar
   * below. Cards with a legal offer are lifted and clicking one commits its single offer or
   * opens its several at the card. The rail repeats every offer by number.
   *
   * The component owns only presentation state (open menus, the preview, where
   * cards were drawn a moment ago, and the revision of the person's last move,
   * which scopes "since your last move"). It commits nothing itself: `onact`
   * receives the exact offer a person chose, and the caller binds it to the
   * frame's revision as one Command. What happened is read from the
   * authority's presentation events; motion only shows it.
   *
   * A recorded game draws the same board read-only (`live` false): the caller
   * supplies where the recap starts (`since`), the move list (`moves`, which
   * may jump), the recorded policy (`evidence`, shown on each offer's card and
   * row) and a bottom edge (`bottom`: the transport and the decision strip).
   *
   * Live play also takes pointer drags (`drag.ts`): each drop goes where a
   * click goes, so a drag adds no choice a click or key cannot make.
   */
  import { onMount, setContext, untrack, type Snippet } from 'svelte';
  import ThemeSwitch from '$lib/components/ThemeSwitch.svelte';
  import type { PreviewState } from '$lib/payment-preview';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import type { CardState, ConnectionState, ExperienceFrame, InteractionOffer, StopsConfig, StopSide } from '$lib/types';
  import Field from './Field.svelte';
  import Inspect from './Inspect.svelte';
  import { permissionTerms } from './inspect';
  import Lines from './Lines.svelte';
  import { animateDeathTravel, collectDeathMarks, prefersReducedMotion, settle, Travel, type DeathMark } from './motion';
  import { carriedAt, cancel as cancelGesture, dropLabel, IDLE, marqueeBounds, move as moveGesture, overlap, playable, press, promote, release, type ConnectDrag, type Gesture, type OfferTarget } from './drag';
  import { region } from './region';
  import { computeStopDeltas, cursorView, hasWalkableTurn, PACE_MS, scrubberStops, sharedLaterCount, stopHasSeq, unreadWalkStops, walkStops, type WalkPace } from './beats';
  import { buildCombatTable } from './combat-table';
  import { findBoard, type StepBoard } from './boards';
  import { buildHistory, cursorReading, hasTrackHistory, recentHistory, stepCursor, turnAt, type CursorScope, type CursorSelection, type HistoryLine, type HistoryNote, type HistoryTurn } from './history';
  import { EVIDENCE, type OfferEvidence, type StripLine } from './replay';
  import { paymentPlan, previewable } from './payment';
  import { addAttackers, addressOf, ATTACK_INSTRUCTION, BLOCK_INSTRUCTION, committedOf, drawDraft, draftWords, EMPTY_DRAFT, isDeclaration, loadDraft, pileCounts as draftPiles, planKey, planOf, saveDraft,
    setBlock, setPileCount, stepOffer, toggleAttacker, toggleBlock, type CombatFetcher, type CombatPreview, type Draft } from './combat';
  import {
    loadPartitionDraft, partitionContext, partitionOffer, partitionSteps, partitionWords,
    savePartitionDraft, swapPiles, toggleCardPile, type PartitionStep,
  } from './partition';
  import { offerText, primaryLabel, promptHint, promptLine } from './prompt';
  import { recentMarks, boardLines, turnOwners, type Endpoint, type RecentMarks } from './recent';
  import Plate from './Plate.svelte';
  import DeckTracker from './DeckTracker.svelte';
  import { buildOpponentTracker, buildSelfTracker, loadTrackerCardFaces, type TrackerCardFace } from './deck-tracker';
  import { autoTapNote, autoTapOffer, nextFullControl, type FullControlMode } from './auto-tap';
  import Rail from './Rail.svelte';
  import BoardCard from './BoardCard.svelte';
  import { phaseTrack, type OnceStops } from './phases';
  import StopBar from './StopBar.svelte';
  import Strip from './Strip.svelte';
  import type { DecisionMark } from '$lib/analysis';
  import { decidesBy, shortName } from '$lib/opponent-name';
  import { CARD_RATIO, fieldItems, footprint, handFan, handStrip, inspectableCards, pileShapes, possessive, splitHalf, boardView, type HalfSplit, type HandItem, type PermanentItem, type PlacedOffer, type RowLimits, type StackItem, type TrayItem } from './view';

  interface Props {
    frame: ExperienceFrame;
    /** Play submits live Commands. Study clicks and keys try an isolated branch. */
    interaction: 'play' | 'study' | 'read-only';
    status: string;
    you: { name: string; deck: string | null };
    opponent: { name: string; deck: string | null; description?: string };
    stops: StopsConfig;
    /** Stops set to hold once (live play). */
    onceStops?: OnceStops;
    /** Cycle one stop from the stop bar; without it the bar is read-only. */
    oncyclestop?: (side: StopSide, step: string) => void;
    /** The live connection; a recording has none. */
    connection?: ConnectionState | null;
    connectionLabel?: string;
    /** Client-local automation/yield notes merged into live history. */
    notes?: readonly HistoryNote[];
    canPassTurn: boolean;
    /** The live presentation track: what happened, as the authority reports it. */
    presentation: PresentationPlayer;
    /** Set while the opponent is choosing, or the server is working, in words. */
    thinking: string | null;
    /** The opponent is the one choosing now. */
    opponentChoosing: boolean;
    /** When the work the game waits on started, epoch seconds; null when nothing runs. */
    workStarted: number | null;
    /** The live session is gone; offers are the last recorded decision's. */
    interrupted: boolean;
    /** The automatic payer's answer for the offer last asked about, if any. */
    payment?: PreviewState | null;
    /** Ask for the payment preview of this offer, or of none. Asking sends no action. */
    onpreview?: (offer: InteractionOffer | null) => void;
    onact: (offer: InteractionOffer) => void;
    /** A Command the person sent is awaiting the authority's answer. */
    pending?: boolean;
    onpassturn?: () => void;
    /** Where "since your last move" starts, when the caller knows (a recording); else the person's last commit. */
    since?: number | null;
    /** The history list, when the caller keeps it (a recording); else read from `presentation`. */
    history?: HistoryTurn[];
    /** Jump to the position a move names; moves are buttons when set. */
    onjump?: (id: string, boardSeq: number | null) => void;
    /** Nothing here can be answered (a recording, or a policy's line through it): the rail drops its answer buttons and empty stack. Defaults to `onjump` being set. */
    readOnly?: boolean;
    /** The move the shown position belongs to. */
    currentMove?: string | null;
    /** External cursor seq (Review sub-step inside a `consequences` transition). */
    cursorSeq?: number | null;
    /** Error reconstructing `stepBoards` patches (surfaces "Board unavailable for this step"). */
    stepBoardsError?: string | null;
    /** Clear Review sub-step cursor (Esc / "Esc for now" / clicking the earlier board). */
    onclearcursor?: () => void;
    /** The recorded policy at a recorded choice. */
    evidence?: OfferEvidence | null;
    /** An analysis's mark on each of your moves, by move id. */
    moveMarks?: ReadonlyMap<string, DecisionMark>;
    /** Your moves chosen with a live analysis shown (Review), by move id. */
    shownMoves?: ReadonlySet<string>;
    /** The recap's heading, when it is not the person's last move. */
    recapTitle?: string;
    /** What the shown position is, when the caller knows (a recording): "After Heuristic's move". */
    heading?: string | null;
    /** Under the board: a recording's transport. */
    bottom?: Snippet;
    /** The game's decisions: drawn under the board while open, its toggle in the stop bar's row. */
    strip?: StripLine;
    notices?: Snippet;
    result?: Snippet;
    /** The Settings sheet's page-owned groups, in the sheet's order: Play, Analysis, (the board's own View), Game. */
    menu?: { play?: Snippet; analysis?: Snippet; game?: Snippet };
    /** Controls of the position shown that stay in the bar (a recording's perspective). */
    bar?: Snippet;
    /** Page settings that are off their default, shown in the bar; pressing one turns it back. */
    chips?: readonly { key: string; label: string; clear: () => void }[];
    historyHref?: string;
    /** The native combat preview, when this table may draft a whole declaration (live play). */
    combatPreview?: CombatFetcher;
    /** The person's board after each step that led to this frame: each beat is shown on its own. */
    stepBoards?: readonly StepBoard[];
    /** Arena-style client Auto-tap: auto-submits mandatory pure-mana payments when Full Control is off. */
    autoTap?: boolean;
    fullControl?: FullControlMode;
    ontoggleautotap?: () => void;
    onfullcontrol?: (mode: FullControlMode) => void;
    onautotap?: (offer: InteractionOffer, note: string) => void;
    /** Walk pace: Study (1.4 s, default), Quick (0.7 s), or Off. */
    walkPace?: WalkPace;
    onwalkpace?: (pace: WalkPace) => void;
  }

  let {
    frame, interaction, status, you, opponent, stops, onceStops, oncyclestop, connection = null, connectionLabel = '', notes = [], canPassTurn,
    presentation, thinking, opponentChoosing, workStarted, interrupted, payment = null, onpreview = () => {}, onact, pending = false, onpassturn = () => {},
    since: sinceGiven, history: historyGiven, onjump, readOnly = onjump !== undefined, currentMove = null, cursorSeq = null, stepBoardsError = null, onclearcursor,
    evidence = null, moveMarks, shownMoves, recapTitle, heading = null, bottom, strip, notices, result, menu, bar, chips = [], historyHref = '/games',
    combatPreview, stepBoards = [],
    autoTap = false, fullControl = 'off', ontoggleautotap, onfullcontrol, onautotap,
    walkPace = 'study', onwalkpace,
  }: Props = $props();

  const live = $derived(interaction !== 'read-only');
  const canDrag = $derived(interaction === 'play');

  // Cards read the recorded policy for their own offers.
  setContext(EVIDENCE, () => evidence);

  // The game fills the window; the page behind it must not scroll.
  // What the page draws behind it is out of reach too: Tab and screen readers
  // stay on the board and the menus and previews it opens.
  onMount(() => {
    const page = document.documentElement;
    const previous = page.style.overflow;
    page.style.overflow = 'hidden';
    const hidden: HTMLElement[] = [];
    for (let node = root?.parentElement; node && node !== document.body; node = node.parentElement) {
      for (const sibling of node.parentElement?.children ?? []) {
        if (sibling !== node && sibling instanceof HTMLElement && !sibling.inert) { sibling.inert = true; hidden.push(sibling); }
      }
    }
    return () => {
      page.style.overflow = previous;
      hidden.forEach(element => { element.inert = false; });
    };
  });

  // Sentences name the opponent shortly; the plate and top bar keep its kind.
  const said = $derived(shortName(opponent.name));
  const players = $derived({ you: shortName(you.name), opponent: said });
  const projection = $derived(frame.projection);
  const names = $derived(new Map([...projection.agent.battlefield, ...projection.opponent.battlefield]
    .map(permanent => [permanent.id, permanent.name ?? 'permanent'])));

  // ———————————————— Dragging

  // The pointer gesture in progress (`drag.ts`). It is drawn only at the
  // revision it started at, so a new frame ends it in the same update.
  let gesture = $state.raw<Gesture>(IDLE);
  const current = $derived(gesture.phase !== 'idle' && gesture.revision === frame.revision ? gesture : IDLE);
  /** The card carried from hand, while it is drawn away from its slot. */
  const carried = $derived((current.phase === 'dragging' || current.phase === 'holding') && current.drag.kind === 'play' ? current.drag : null);
  const carriedPoint = $derived(carriedAt(current));
  const dragging = $derived(current.phase === 'dragging');

  const connect = $derived(current.phase === 'dragging' && current.drag.kind === 'connect' ? current.drag : null);
  const snapped = $derived(connect?.target?.endpoint);
  const marquee = $derived(current.phase === 'dragging' && current.drag.kind === 'marquee' ? marqueeBounds(current.start, current.pointer) : null);
  const draftLine = $derived(connect && current.phase === 'dragging'
    ? { from: connect.from, to: snapped ?? current.pointer, kind: connect.style } : null);

  // ———————————————— Combat draft

  // At an attack or block declaration a person drafts the whole declaration
  // on the board. The native preview (read-only) says which creatures may
  // take part and returns the exact per-creature offers that declare the
  // draft; nothing is sent until Confirm, which sends them one at a time and
  // stops at any frame the preview did not predict.
  let draft = $state.raw<Draft>(EMPTY_DRAFT);
  /** The blocker a person is placing: the next attacker clicked is the one it blocks. */
  let armed = $state<number | null>(null);
  let previewed = $state.raw<{ frameHash: string; plan: string; preview: CombatPreview } | null>(null);
  /** Confirmed steps being sent: `at` was sent at frame `sentAt`. */
  let running = $state.raw<{ preview: CombatPreview; at: number; sentAt: number } | null>(null);
  let stopNote = $state<string | null>(null);
  let draftFrame = '';
  const declaring = $derived(combatPreview !== undefined && isDeclaration(frame));
  const preview = $derived(running?.preview ?? (previewed?.frameHash === frame.frame_hash ? previewed.preview : null));
  const drafting = $derived(declaring && !running && preview !== null && preview.declarable.length > 0);
  const joint = $derived(drafting || running !== null);
  const plan = $derived(preview ? planOf(draft, preview) : null);
  /** The preview answers this very draft, not an earlier one. */
  const exact = $derived(preview !== null && plan !== null && previewed?.plan === planKey(plan) ? preview : null);

  // A new declaration starts from the draft saved for it in this tab, if any.
  $effect(() => {
    // Live play may connect after the frame is drawn: the draft loads once the board may declare.
    const key = `${frame.match_id}:${frame.frame_hash}:${declaring}`;
    if (key === draftFrame) return;
    draftFrame = key;
    untrack(() => {
      armed = null;
      if (running) return;
      draft = declaring ? loadDraft(frame) ?? EMPTY_DRAFT : EMPTY_DRAFT;
    });
  });
  // Every draft change asks the preview again; an answer for another frame or draft is dropped.
  $effect(() => {
    const fetcher = combatPreview;
    const address = addressOf(frame);
    const current = draft;
    if (!fetcher || !declaring || !address || running) return;
    // The first answer names the creatures; a draft restored before it is asked again with their identities.
    const known = previewed?.frameHash === frame.frame_hash ? previewed : null;
    const request = known ? planOf(current, known.preview) : { attackers: [], blocks: [] };
    if (known && known.plan === planKey(request)) return;
    const controller = new AbortController();
    const frameHash = frame.frame_hash;
    const actor = frame.prompt?.actor;
    fetcher({ ...address, plan: request }, controller.signal).then(response => {
      if (controller.signal.aborted) return;
      if (Object.entries(address).some(([name, value]) => response[name as keyof typeof address] !== value) || response.viewer !== actor) return;
      previewed = { frameHash, plan: planKey(request), preview: response.preview };
    }).catch(() => { /* Without a preview the board keeps one creature at a time. */ });
    return () => controller.abort();
  });
  $effect(() => {
    if (drafting) saveDraft(frame, draft);
  });

  const words = $derived(preview ? draftWords(preview.kind, draft, preview.declarable, names, committedOf(frame)) : null);
  const declarable = $derived(new Map((preview?.declarable ?? []).map(creature => [creature.creature.entity, creature])));
  const draggedBlocker = $derived(connect?.blocker);
  const placingBlocker = $derived(draggedBlocker ?? armed);
  /** Each creature's place in the draft and what pressing it does. */
  const draftMarks = $derived.by(() => {
    if (!joint || !preview) return null;
    const places = new Map<number, 'can' | 'in' | 'armed'>();
    const notes = new Map<number, string>();
    if (preview.kind === 'attackers') {
      for (const [id, creature] of declarable) {
        const drafted = draft.attackers.includes(id);
        places.set(id, drafted ? 'in' : 'can');
        notes.set(id, drafted ? 'in the drafted attack; press to leave it out'
          : creature.must_attack ? 'must attack; press to add it to the attack' : 'can attack; press to add it to the attack');
      }
    } else {
      const blocks = new Map(draft.blocks);
      for (const [id, creature] of declarable) {
        const blocking = blocks.get(id);
        places.set(id, placingBlocker === id ? 'armed' : blocking !== undefined ? 'in' : 'can');
        notes.set(id, placingBlocker === id ? 'placing this block; press an attacker it can block'
          : blocking !== undefined ? `drafted to block ${names.get(blocking) ?? 'an attacker'}; press to move or take back the block`
            : creature.must_block ? 'must block; press, then the attacker it blocks' : 'can block; press, then the attacker it blocks');
      }
      const placing = placingBlocker === null ? null : declarable.get(placingBlocker);
      for (const attacker of placing?.can_block ?? []) {
        const id = attacker.entity;
        const taken = draft.blocks.some(([b, a]) => b === placingBlocker && a === id);
        places.set(id, 'can');
        notes.set(id, `${taken ? 'take back the block by' : 'block it with'} ${names.get(placingBlocker!) ?? 'the blocker'}`);
      }
    }
    return { places, notes };
  });
  /** Identical creatures drawn as one pile take a count: "4 of 9 Allies". */
  const counts = $derived.by(() => {
    if (!drafting || !preview || preview.kind !== 'attackers') return [];
    // Piles as drawn without offers: the creature asked about now is still one of its pile.
    const base = boardView({ ...frame, offers: [] }, players);
    const pileOf = new Map(base.you.creatures.flatMap(pile => pile.items.map(item => [item.permanent.id, pile.key] as const)));
    return draftPiles(preview.declarable, draft, id => pileOf.get(id) ?? null);
  });

  function onDraft(id: number): boolean {
    if (!live || !drafting || !preview) return false;
    stopNote = null;
    if (preview.kind === 'attackers') {
      if (!declarable.has(id)) return false;
      draft = toggleAttacker(draft, id);
      return true;
    }
    const blocker = declarable.get(id);
    if (blocker) {
      // A blocker with exactly one attacker to block takes it at once.
      if (blocker.can_block.length === 1 && armed === null) { draft = toggleBlock(draft, id, blocker.can_block[0].entity); return true; }
      armed = armed === id ? null : id;
      return true;
    }
    if (armed !== null && declarable.get(armed)?.can_block.some(attacker => attacker.entity === id)) {
      draft = toggleBlock(draft, armed, id);
      armed = null;
      return true;
    }
    return false;
  }

  function confirmDraft(): void {
    const ready = exact;
    if (!live || !drafting || !ready || ready.refused || !ready.steps.length) return;
    const offer = stepOffer(frame, ready.kind, ready.steps[0]);
    if (!offer) { stopNote = 'The board changed before the plan was sent; draft again.'; return; }
    running = { preview: ready, at: 0, sentAt: frame.revision };
    send(offer);
  }

  // Each frame after a sent step must be the decision the preview predicted for the next. The accepted
  // step's frame can arrive before the table takes input again; the next step waits for it rather than
  // judging a frame no one may answer yet.
  $effect(() => {
    const revision = frame.revision;
    const sending = running;
    if (!sending) return;
    if (!combatPreview) { running = null; draft = EMPTY_DRAFT; return; }
    const answerable = live && !pending;
    const { steps, kind } = sending.preview;
    untrack(() => {
      if (revision === sending.sentAt) {
        if (!answerable) return;
        running = null;
        if (sending.at > 0) {
          draft = EMPTY_DRAFT;
          stopNote = `Stopped after ${sending.at} of ${steps.length}. The next declaration was not accepted. Draft again from here.`;
        }
        return;
      }
      const next = sending.at + 1;
      if (next >= steps.length) { running = null; draft = EMPTY_DRAFT; return; }
      if (!answerable) return;
      const offer = stepOffer(frame, kind, steps[next]);
      if (!offer) {
        running = null;
        draft = EMPTY_DRAFT;
        stopNote = `Stopped after ${next} of ${steps.length}: the game asked something the preview did not predict. The board shows where it stands; draft again from here.`;
        return;
      }
      running = { ...sending, at: next, sentAt: revision };
      onact(offer);
    });
  });
  function stopSending(): void {
    if (!running) return;
    stopNote = `Stopped after ${running.at + 1} of ${running.preview.steps.length}; the rest was not sent.`;
    running = null;
    draft = EMPTY_DRAFT;
  }

  // ———————————————— Pile partition draft

  // Curator of Destinies asks about one inspected card at a time. Show all
  // remaining inspected cards together so a person can split them holistically,
  // then send the engine's exact per-card offers one by one on Confirm.
  let partitionFaceUp = $state.raw<number[]>([]);
  let partitionRunning = $state.raw<{ steps: PartitionStep[]; at: number; sentAt: number } | null>(null);
  let partitionFolded = $state(false);
  let partitionFrameKey = '';
  const partition = $derived(live && !drafting && !running ? partitionContext(frame) : null);
  const partitioning = $derived(partition !== null && !partitionRunning);

  $effect(() => {
    const active = partition !== null;
    const key = `${frame.match_id}:${frame.frame_hash}:${active}`;
    if (key === partitionFrameKey) return;
    partitionFrameKey = key;
    untrack(() => {
      if (partitionRunning) return;
      partitionFolded = false;
      if (!partition) { partitionFaceUp = []; return; }
      const validIds = new Set(partition.remaining.map(card => card.id));
      partitionFaceUp = loadPartitionDraft(frame, validIds) ?? [];
    });
  });
  $effect(() => {
    if (partitioning) savePartitionDraft(frame, partitionFaceUp);
  });

  const partitionSummary = $derived(partition ? partitionWords(partition.remaining, partitionFaceUp, partition.committed.length) : null);
  const partitionUpSet = $derived(new Set(partitionFaceUp));
  const partitionUpCards = $derived(partition ? partition.remaining.filter(card => partitionUpSet.has(card.id)) : []);
  const partitionDownCards = $derived(partition ? partition.remaining.filter(card => !partitionUpSet.has(card.id)) : []);

  function togglePartitionCard(cardId: number): void {
    if (!partitioning) return;
    stopNote = null;
    partitionFaceUp = toggleCardPile(partitionFaceUp, cardId);
  }

  function confirmPartition(): void {
    if (!partitioning || !partition || !partition.remaining.length) return;
    const steps = partitionSteps(partition.remaining, partitionFaceUp);
    const offer = partitionOffer(frame, steps[0]);
    if (!offer) {
      stopNote = 'The decision changed before the split was sent; split the remaining cards again.';
      return;
    }
    partitionRunning = { steps, at: 0, sentAt: frame.revision };
    send(offer);
  }

  $effect(() => {
    const revision = frame.revision;
    const sending = partitionRunning;
    if (!sending) return;
    const answerable = live && !pending;
    untrack(() => {
      if (revision === sending.sentAt) {
        if (!answerable) return;
        partitionRunning = null;
        if (sending.at > 0) {
          partitionFaceUp = [];
          stopNote = `Stopped after ${sending.at} of ${sending.steps.length}. The next choice was not accepted. Split any remaining cards from here.`;
        }
        return;
      }
      const next = sending.at + 1;
      if (next >= sending.steps.length) { partitionRunning = null; partitionFaceUp = []; return; }
      if (!answerable) return;
      const offer = partitionOffer(frame, sending.steps[next]);
      if (!offer) {
        partitionRunning = null;
        partitionFaceUp = [];
        stopNote = `Stopped after ${next} of ${sending.steps.length}: the game asked something unexpected. The board shows where it stands; split any remaining cards from here.`;
        return;
      }
      partitionRunning = { ...sending, at: next, sentAt: revision };
      onact(offer);
    });
  });

  function stopPartitionSending(): void {
    if (!partitionRunning) return;
    stopNote = `Stopped after ${partitionRunning.at + 1} of ${partitionRunning.steps.length}; the rest was not sent.`;
    partitionRunning = null;
    partitionFaceUp = [];
  }

  /** The frame as the board draws it: with the drafted declaration, while there is one. */
  const drafted = $derived(joint ? drawDraft(frame, draft) : frame);
  const view = $derived(boardView(drafted, players));
  const whose = $derived(heading ?? (frame.status === 'game_over' ? 'Game over'
    : !frame.prompt ? 'Waiting'
      : frame.prompt.actor === projection.agent.player_index
        ? you.name === 'You' ? (live ? 'Your decision' : 'Your decision · waiting') : `${possessive(you.name)} decision`
        : `${possessive(said)} decision`));
  const targeting = $derived(canDrag && !joint && (frame.action_space === 'CHOOSE_TARGET'
    || frame.action_space === 'DECLARE_BLOCKER' || frame.offers.some(offer => offer.subjects?.some(s => s.role === 'target'))));
  const fieldSources = $derived.by(() => {
    if (canDrag && drafting && preview?.kind === 'blockers') return new Set(declarable.keys());
    if (!targeting) return new Set<number>();
    const ids = new Set(view.asked);
    // A forced block can have only attacker offers, without a decline button naming the blocker.
    if (frame.action_space === 'DECLARE_BLOCKER') {
      for (const { offer } of view.list) {
        if (offer.verb !== 'declare_blockers') continue;
        for (const { role, object } of offer.subjects ?? []) {
          if (role === 'choice' && object.subject.kind === 'object') ids.add(object.display_id);
        }
      }
    }
    return ids;
  });
  const stackSources = $derived.by(() => {
    const source = projection.decision_context?.source;
    const ids = new Set<number>();
    if (!targeting) return ids;
    for (const item of view.stack) {
      const asked = source?.subject.kind === 'stack' ? item.entry.id === source.subject.id
        : source?.subject.kind === 'object' && item.card?.id === source.display_id;
      if (item.announcing || asked) ids.add(item.entry.id);
    }
    return ids;
  });
  const fieldById = $derived(new Map([...fieldItems(view.you), ...fieldItems(view.opponent)].map(item => [item.permanent.id, item])));
  // Keep the recipient list steady as the pointer moves between them.
  const connecting = $derived(connect !== null);
  const connectTargets = $derived.by(() => {
    if (!connecting) return undefined;
    if (draggedBlocker !== undefined) return new Set(declarable.get(draggedBlocker)?.can_block.map(attacker => attacker.entity));
    const targets = new Set<number>();
    for (const [id, item] of fieldById) if (item.offers.length) targets.add(id);
    return targets;
  });
  const stackById = $derived(new Map(view.stack.map(item => [item.entry.id, item])));

  // ———————————————— Payment preview

  // The offer the board previews: the one a person points at or focuses, when
  // the authority can preview it (a cast, an activation, a tap), else the
  // pending payment itself. Target choices highlight the candidate target on
  // the board instead of distracting with paying lands.
  let pointed = $state<PlacedOffer | null>(null);
  const payOffer = $derived(view.primary?.offer.verb === 'pay' ? view.primary : null);
  const pointedTarget = $derived.by<Endpoint | null>(() => {
    if (!pointed || !live || activeCursor !== null) return null;
    if (pointed.anchor.kind === 'object') return { kind: 'object', id: pointed.anchor.id };
    if (pointed.anchor.kind === 'player') return { kind: 'player', index: pointed.anchor.index };
    if (pointed.anchor.kind === 'stack') return { kind: 'stack', id: pointed.anchor.id };
    return null;
  });
  const pointedIsTargetChoice = $derived(pointed?.offer.action_type === 'CHOOSE_TARGET');
  const pointedObjectIds = $derived(pointedTarget?.kind === 'object' ? new Set([pointedTarget.id]) : new Set<number>());
  const pointedPlayerIndex = $derived(pointedTarget?.kind === 'player' ? pointedTarget.index : null);
  // Pointing at the pending payment's own row is still that payment: the
  // prompt keeps its plan and note, so nothing shifts under the pointer.
  // A card carried toward the battlefield previews its single offer for the whole drag, touch included.
  const dragOffers = $derived(carried?.offers ?? (connect?.blocker === undefined ? connect?.target?.offers ?? null : null));
  const dragOffer = $derived(dragOffers?.length === 1 && previewable(dragOffers[0].offer) ? dragOffers[0] : null);
  const previewing = $derived(!live ? null : dragOffer
    ?? (!dragging && pointed && !pointedIsTargetChoice && previewable(pointed.offer) && pointed.offer.id !== payOffer?.offer.id ? pointed : payOffer));
  $effect(() => { onpreview(previewing?.offer ?? null); });
  const answered = $derived(payment && previewing && payment.offer === previewing.offer.id && !payment.loading ? payment.preview : null);
  const payPlanned = $derived(paymentPlan(answered, frame));
  // Piles part around whatever plan is shown, so each planned land has its own
  // outline and mark. The pending payment's own answer is kept for its
  // revision: piles, Pay label and note read it while a person points
  // at other choices, so pointing never reflows the rail or moves a card from
  // under the pointer (which would change what is pointed at).
  let payAnswer = $state.raw<{ frameHash: string; offerId: number; answered: NonNullable<typeof answered> } | null>(null);
  $effect(() => {
    if (payOffer && previewing === payOffer && answered) {
      payAnswer = { frameHash: frame.frame_hash, offerId: payOffer.offer.id, answered };
    }
  });
  const payKnown = $derived(live && payAnswer?.frameHash === frame.frame_hash && payAnswer.offerId === payOffer?.offer.id
    ? payAnswer.answered : null);
  const payKnownPlan = $derived(paymentPlan(payKnown, frame));
  const unpiled = $derived((payKnownPlan ?? payPlanned)?.sources ?? null);
  const primaryText = $derived(primaryLabel(frame, view, payKnown));
  const line = $derived(drafting && words ? (exact?.refused ? exact.refused.reason : words.plan)
    : running ? `Declaring ${running.at + 1} of ${running.preview.steps.length}…`
      : partitioning && partitionSummary ? partitionSummary.plan
        : partitionRunning ? `Placing ${partitionRunning.at + 1} of ${partitionRunning.steps.length}…`
          : promptLine(frame, view, !readOnly));
  // How to answer with the live input; a recording has none.
  const hint = $derived(joint || partitioning || partitionRunning || readOnly ? null : promptHint(frame, view));
  const paymentNote = $derived(!payOffer ? null
    : payment?.error && previewing === payOffer ? 'The payment preview is unavailable; Pay still pays automatically.'
      : payKnown && payKnown.state !== 'exact' && payKnown.state !== 'not_applicable' ? payKnown.explanation : null);
  function point(offers: PlacedOffer[] | null): void {
    if (dragging) return;
    pointed = offers?.length === 1 ? offers[0] : null;
  }

  // ———————————————— What happened since your last move

  // The revision your last move was made at; events after it are "since your last move".
  let since = $state.raw<number | null>(null);
  const sinceRevision = $derived(sinceGiven !== undefined ? sinceGiven : since);
  const recent = $derived(recentMarks(frame, sinceRevision, presentation.events, index => presentation.labelsAt(index)));

  // ———————————————— History and one cursor

  const tellPlayers = $derived({ youSeat: projection.agent.player_index, you: players.you, opponent: players.opponent });
  const history = $derived(historyGiven ?? buildHistory(
    presentation.events,
    index => presentation.labelsAt(index),
    tellPlayers,
    {
      notes,
      initialTurn: projection.turn.turn_number,
      initialOwner: projection.turn.active_player_id === projection.agent.id ? 'you' : 'opponent',
    },
  ));

  // One cursor into the history (`selection: { seq, scope } | null` + timer
  // state `walking: boolean`). `selection === null` rests on now. Walking,
  // clicking a scrubber tick, and "Replay turn" select `scope: 'stop'`
  // (keeping multi-line stops intact when paused); clicking a history line,
  // pressing ← →, or stepping into a Review transition selects `scope: 'line'`.
  let shownSeq = untrack(() => presentation.events.at(-1)?.seq ?? -1);
  let shownMatch = untrack(() => frame.match_id);
  let shownRevision = untrack(() => frame.revision);
  let wasConnected = untrack(() => connection === 'connected');
  let selection = $state<CursorSelection | null>(null);
  let walking = $state(false);
  let walkQueue = $state.raw<number[]>([]);
  let captionHovered = $state(false);
  let combatTableHovered = $state(false);
  let combatRowSubjects = $state<number[]>([]);

  const activePaceMs = $derived(PACE_MS[walkPace]);

  const activeCursor = $derived<CursorSelection | null>(
    connection !== null ? selection : cursorSeq !== null ? { seq: cursorSeq, scope: 'line' } : null,
  );
  const cursor = $derived(activeCursor?.seq ?? null);

  const stopsAll = $derived(walkStops(history, stepBoards));
  const scrubber = $derived(scrubberStops(stopsAll, cursor));
  const hasWalkable = $derived(hasWalkableTurn(stopsAll));
  const showWalkBar = $derived(connection !== null && (hasWalkable || selection !== null));
  const active = $derived(cursorView(history, stopsAll, activeCursor));
  const activeStopIndex = $derived(cursor === null ? -1 : scrubber.stops.findIndex(stop => stopHasSeq(stop, cursor)));

  function blurHistoryControl(): void {
    const focused = typeof document !== 'undefined' ? document.activeElement : null;
    if (focused instanceof HTMLElement && focused.closest('.log, .walk-bar')) {
      focused.blur();
    }
  }

  function setCursor(next: number | null, scope: CursorScope = 'line'): void {
    walking = false;
    walkQueue = [];
    captionHovered = false;
    combatTableHovered = false;
    combatRowSubjects = [];
    selection = next === null ? null : { seq: next, scope };
    if (next === null) blurHistoryControl();
  }

  function returnToNow(): void {
    setCursor(null);
  }

  function selectHistoryLine(selected: HistoryLine): void {
    const seq = selected.boardSeq ?? selected.seq;
    if (seq !== null) setCursor(seq, 'line');
  }

  function startWalk(seqs: number[], auto: boolean): void {
    walkQueue = seqs;
    selection = { seq: seqs[0], scope: 'stop' };
    captionHovered = false;
    combatTableHovered = false;
    combatRowSubjects = [];
    walking = auto;
  }

  function replayTurn(): void {
    if (!scrubber.stops.length) return;
    startWalk(
      scrubber.stops.map(stop => stop.seq),
      !prefersReducedMotion() && activePaceMs !== null,
    );
  }

  // Before the DOM updates: a frame that brings unread stops starts walking
  // on the first stop's board instead of flashing the final frame. If a walk
  // is already running, newly arrived stops append to its queue without
  // cutting the current stop short or aborting on an empty frame.
  $effect.pre(() => {
    const currentStops = stopsAll;
    const lastSeq = presentation.events.at(-1)?.seq ?? -1;
    const { match_id, revision } = frame;
    const connected = connection === 'connected';
    const ms = activePaceMs;
    if (connection === null) return;
    untrack(() => {
      if (!connected || !wasConnected || match_id !== shownMatch || revision < shownRevision) {
        returnToNow();
        shownSeq = lastSeq;
      } else if (revision === shownRevision) {
        // Recovery filled in history the table had already shown before a
        // reload: those facts are read, never walked again.
        shownSeq = Math.max(shownSeq, lastSeq);
      } else {
        const fresh = prefersReducedMotion() || ms === null ? [] : unreadWalkStops(currentStops, shownSeq);
        if (walking && selection !== null) {
          if (fresh.length > 0) {
            const existing = new Set(walkQueue);
            const appended = fresh.map(stop => stop.seq).filter(seq => !existing.has(seq));
            if (appended.length > 0) walkQueue = [...walkQueue, ...appended];
          }
        } else if (fresh.length > 0) {
          startWalk(fresh.map(stop => stop.seq), true);
        }
        shownSeq = Math.max(shownSeq, lastSeq);
      }
      shownMatch = match_id;
      shownRevision = revision;
      wasConnected = connected;
    });
  });

  // Walk timer: advances one kept board stop per pace tick; pauses while the
  // pointer rests on the caption or combat table. Reads `walkQueue` untracked
  // inside the timeout so appending newly arrived stops never restarts the
  // current stop's beat.
  $effect(() => {
    const seq = selection?.seq ?? null;
    const ms = activePaceMs;
    if (!walking || seq === null || captionHovered || combatTableHovered || ms === null) return;
    const timer = setTimeout(() => {
      untrack(() => {
        if (!walking || selection?.seq !== seq) return;
        const index = walkQueue.indexOf(seq);
        if (index >= 0 && index + 1 < walkQueue.length) {
          selection = { seq: walkQueue[index + 1], scope: 'stop' };
        } else {
          returnToNow();
        }
      });
    }, ms);
    return () => clearTimeout(timer);
  });

  // Window totals, now's `.touched` rings, the recap, and the result card wait until the cursor reaches now.
  const NO_DAMAGE: RecentMarks['damage'] = { objects: new Map(), players: new Map() };
  const EMPTY_TOUCHED = new Set<number>();
  const damage = $derived(activeCursor !== null ? NO_DAMAGE : recent.damage);
  const touched = $derived(activeCursor !== null ? EMPTY_TOUCHED : recent.touched);
  const finished = $derived(result !== undefined && activeCursor === null);
  const recapSource = $derived(
    historyGiven && !hasTrackHistory(presentation.events)
      ? buildHistory(presentation.events, index => presentation.labelsAt(index), tellPlayers)
      : history,
  );
  const recentFacts = $derived(
    activeCursor !== null ? [] : recentHistory(recapSource, sinceRevision, presentation.events.at(-1)?.seq ?? null),
  );

  // Look up the kept board for `activeCursor`: a past board has no live offers,
  // while the rail stays live throughout.
  const cursorHit = $derived(activeCursor !== null && !stepBoardsError ? findBoard(stepBoards, activeCursor.seq) : null);
  const boardMissing = $derived(activeCursor !== null && !stepBoardsError && cursorHit === null);
  const boardSharedCount = $derived(activeCursor !== null && cursorHit?.fallback ? sharedLaterCount(history, activeCursor.seq, cursorHit.last) : 0);
  const drawn = $derived(activeCursor !== null ? { ...frame, projection: cursorHit?.projection ?? frame.projection, offers: [] } : drafted);
  const board = $derived(activeCursor !== null ? boardView(drawn, players) : view);
  const boardProjection = $derived(drawn.projection);
  const priorBoardProjection = $derived.by(() => {
    if (activeCursor === null) return null;
    const minSeq = Math.min(...active.seqs, activeCursor.seq);
    for (let i = stepBoards.length - 1; i >= 0; i -= 1) {
      if (stepBoards[i].first < minSeq) return stepBoards[i].projection;
    }
    return null;
  });
  const stopDeltas = $derived(computeStopDeltas(presentation.events, active.seqs, boardProjection));
  const combatTable = $derived(
    activeCursor !== null
      ? buildCombatTable(
          presentation.events,
          active.seqs,
          index => presentation.labelsAt(index),
          tellPlayers,
          boardProjection,
          priorBoardProjection,
        )
      : null,
  );
  const activeSubjects = $derived(
    combatRowSubjects.length ? new Set([...active.subjects, ...combatRowSubjects]) : active.subjects,
  );
  const owners = $derived(turnOwners(frame, presentation.events));
  const turnNumber = $derived(activeCursor !== null ? turnAt(presentation.events, activeCursor.seq) ?? boardProjection.turn.turn_number : boardProjection.turn.turn_number);
  const turnStep = $derived(turnNumber === boardProjection.turn.turn_number ? boardProjection.turn.step : null);
  const yourTurn = $derived(turnStep !== null ? boardProjection.turn.active_player_id === boardProjection.agent.id : owners.get(turnNumber) === 'you');
  const track = $derived(phaseTrack(turnStep, stops, yourTurn, onceStops));
  const lines = $derived(boardLines(drawn, presentation.events, activeCursor !== null ? [] : recent.resolved, activeCursor !== null ? null : sinceRevision, pointedTarget));
  const standingLines = $derived(draggedBlocker === undefined ? lines : lines.filter(line =>
    !(line.kind === 'block' && line.from.kind === 'object' && line.from.id === draggedBlocker)));
  /** Permanents something on the stack targets, or currently pointed at as a candidate target. */
  const targeted = $derived(new Set([
    ...lines.flatMap(line => line.kind === 'target' && line.to.kind === 'object' ? [line.to.id] : []),
    ...pointedObjectIds,
  ]));
  const boardAsked = $derived(
    joint
      ? new Set<number>()
      : pointedObjectIds.size
        ? new Set([...board.asked, ...pointedObjectIds])
        : board.asked,
  );
  /** The person may act on the drawn board now: what they cannot act on steps back. */
  const acting = $derived(live && activeCursor === null && frame.prompt?.actor === projection.agent.player_index && (view.list.length > 0 || drafting || partitioning));
  const boardNotice = $derived.by<{ testid: string; text: string } | null>(() => {
    if (activeCursor === null) return null;
    if (stepBoardsError) return { testid: 'board-unavailable', text: 'Board unavailable for this step' };
    if (boardMissing) return { testid: 'board-not-kept', text: 'Board not kept for this step' };
    if (boardSharedCount > 0) {
      return { testid: 'board-shared', text: `shared with ${boardSharedCount} later ${boardSharedCount === 1 ? 'line' : 'lines'}` };
    }
    return null;
  });
  const activeCursorReading = $derived.by<string | null>(() => {
    if (activeCursor === null) return null;
    const reading = cursorReading(history, presentation.events, activeCursor, players, active.stop);
    return boardNotice && (stepBoardsError || !showWalkBar) ? `${reading} · ${boardNotice.text}` : reading;
  });
  function clearActiveCursor(): void {
    if (selection !== null) returnToNow();
    else {
      blurHistoryControl();
      onclearcursor?.();
    }
  }
  /** The click that ends a drag is the drop, not a press of what it lands on. */
  let swallowClick = false;
  function onPress(event: PointerEvent): void {
    if (event.button !== 0 || !event.isPrimary) return;
    swallowClick = false;
    if (!canDrag || activeCursor !== null || !(event.target instanceof Element)) return;
    const start = { x: event.clientX, y: event.clientY };
    if (canMarquee && event.target.closest('.board .rows')) {
      gesture = press({ kind: 'marquee' }, event.pointerId, frame.revision, start);
      return;
    }
    const source = event.target.closest<HTMLElement>('[data-connect-source]');
    if (!source) return;
    const id = Number(source.dataset.stackEntry ?? source.dataset.objectId);
    const drag: ConnectDrag = source.dataset.stackEntry !== undefined
      ? { kind: 'connect', from: { kind: 'entry', id }, style: 'target', target: null }
      : drafting
        ? { kind: 'connect', from: { kind: 'object', id }, style: 'block', blocker: id, target: null }
        : { kind: 'connect', from: { kind: 'object', id }, style: frame.action_space === 'DECLARE_BLOCKER' ? 'block' : 'target', target: null };
    gesture = press(drag, event.pointerId, frame.revision, start);
  }
  function onClick(event: MouseEvent): void {
    // A keyboard click (detail 0) is never the end of a drag.
    if (swallowClick && event.detail > 0) {
      swallowClick = false;
      event.preventDefault();
      event.stopPropagation();
    }
  }
  // Capturing listeners on the whole game start connect/marquee drags and swallow post-drop clicks.
  $effect(() => {
    const element = root;
    if (!element) return;
    element.addEventListener('pointerdown', onPress, { capture: true });
    element.addEventListener('click', onClick, { capture: true });
    return () => {
      element.removeEventListener('pointerdown', onPress, { capture: true });
      element.removeEventListener('click', onClick, { capture: true });
    };
  });

  // ———————————————— Motion

  let root = $state<HTMLElement | null>(null);
  let boardEl = $state<HTMLElement | null>(null);
  const travel = new Travel();
  let deathMarks = $state.raw<DeathMark[]>([]);

  function runDeathTravel(element: HTMLElement, mark: DeathMark): { destroy: () => void } {
    const animation = animateDeathTravel(element, mark);
    return { destroy: () => { animation.cancel(); } };
  }

  // Cards travel between frames and between the boards stops are shown on.
  $effect.pre(() => {
    void boardProjection;
    travel.snapshot(root);
  });
  $effect(() => {
    const reduced = prefersReducedMotion();
    travel.play(root, reduced);
    deathMarks = collectDeathMarks({
      root,
      boardEl,
      travel,
      projection: boardProjection,
      stepBoards,
      events: presentation.events,
      labelsAt: index => presentation.labelsAt(index),
      activeSeqs: active.seqs,
      cursor,
      sinceRevision,
      matchId: frame.match_id,
      revision: frame.revision,
      reduced,
    });
  });
  const opponentDeaths = $derived(deathMarks.filter(mark => mark.side === 'opponent'));
  const yourDeaths = $derived(deathMarks.filter(mark => mark.side === 'you'));

  // Battlefield frames are printed cards, upright or tapped on their side;
  // lands and other support permanents are smaller cards of the same frame.
  const CREATURES: RowLimits = { gap: 8, minCard: 48, maxCard: 140, maxLines: 3 };
  const SUPPORT: RowLimits = { gap: 6, minCard: 40, maxCard: 104, maxLines: 2 };
  const ROW_GAP = 6;

  // Each half divides its height between its two rows by what they hold.
  let theirRows = $state({ width: 0, height: 0 });
  let yourRows = $state({ width: 0, height: 0 });
  function split(side: typeof view.you, size: { width: number; height: number }, deathCount: number): HalfSplit | null {
    if (size.width <= 0 || size.height <= 0) return null;
    const creatureShapes = pileShapes(side.creatures);
    for (let i = 0; i < deathCount; i += 1) creatureShapes.push([false]);
    return splitHalf(creatureShapes, pileShapes(side.support), size.width - 22, size.height - 16,
      { gap: ROW_GAP, creatures: CREATURES, support: SUPPORT });
  }
  const theirSplit = $derived(split(board.opponent, theirRows, opponentDeaths.length));
  const yourSplit = $derived(split(board.you, yourRows, yourDeaths.length));
  const rowsStyle = (half: HalfSplit | null, size: { height: number }, creaturesFirst: boolean): string | undefined => {
    if (!half) return undefined;
    const rest = Math.max(0, size.height - 12 - ROW_GAP - half.creatures);
    return creaturesFirst ? `${half.creatures}px ${rest}px` : `${rest}px ${half.creatures}px`;
  };

  // ———————————————— Menus and preview

  type Menu = { title: string; offers: PlacedOffer[]; x: number; y: number };
  type Zone = { title: string; cards: CardState[]; x: number; y: number };
  /** What is inspected: a name, and the object it names when there is one. */
  type Subject = { name: string; text: string | null; permanent: number | null; card: number | null };
  let menuAt = $state<Menu | null>(null);
  let zoneAt = $state<Zone | null>(null);
  let menuElement = $state<HTMLElement | null>(null);

  // Inspecting: a preview floats beside the card under the pointer or focus,
  // or the sidebar's docked pane keeps the last card inspected.
  const INSPECT_KEY = 'academy.table.inspect';
  let sheet = $state<HTMLDetailsElement | null>(null);
  let inspectMode = $state<'float' | 'dock'>(typeof localStorage !== 'undefined' && localStorage.getItem(INSPECT_KEY) === 'dock' ? 'dock' : 'float');
  function toggleInspect(): void {
    inspectMode = inspectMode === 'dock' ? 'float' : 'dock';
    localStorage.setItem(INSPECT_KEY, inspectMode);
  }
  let hovered = $state<{ subject: Subject; x: number; y: number } | null>(null);
  let resultFolded = $state(false);
  $effect(() => { if (!result) resultFolded = false; });
  let docked = $state<Subject | null>(null);

  // A new revision replaces the cards a menu or preview was opened from. A
  // frame that repeats the revision (a resume, a stops echo) keeps them, so
  // a resting pointer keeps its preview.
  const revision = $derived(frame.revision);
  $effect(() => {
    void revision;
    menuAt = null;
    hovered = null;
    pointed = null;
    untrack(stopDwell);
  });
  $effect(() => {
    if (menuAt) menuElement?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
  });

  function act(placed: PlacedOffer): void {
    send(placed.offer);
  }
  function send(offer: InteractionOffer): void {
    if (!live) return;
    returnToNow();
    gesture = promote(gesture, offer.id);
    menuAt = null;
    stopNote = null;
    hush();
    since = frame.revision;
    onact(offer);
  }

  function passTurn(): void {
    if (!canPassTurn) return;
    returnToNow();
    since = frame.revision;
    onpassturn();
  }

  /** The control a card menu was opened from; closing the menu returns focus to it. */
  let opener: HTMLElement | null = null;
  function closeMenu(): void {
    if (!menuAt) return;
    menuAt = null;
    if (opener?.isConnected) opener.focus({ preventScroll: true });
    opener = null;
  }
  /** Commit a single offer, or open several beside `element` (or at `at`, where a card was dropped). */
  function activate(title: string, offers: PlacedOffer[], element: HTMLElement, at?: DOMRect): void {
    hush();
    if (!live || !offers.length) return;
    if (offers.length === 1) { act(offers[0]); return; }
    opener = element;
    const rect = at ?? element.getBoundingClientRect();
    menuAt = { title, offers, x: Math.max(8, Math.min(rect.left, window.innerWidth - 300)), y: rect.bottom + 6 > window.innerHeight - 160 ? Math.max(8, rect.top - 8 - 36 * offers.length - 30) : rect.bottom + 6 };
  }

  // Inspect opens only when asked: after the pointer rests on a card for a
  // moment, or at once for keyboard focus. After a commit the pointer stays
  // quiet until it moves, so the card that slides under a resting pointer
  // opens nothing; any frame change closes it.
  const DWELL_MS = 380;
  let dwell: ReturnType<typeof setTimeout> | null = null;
  let pointer = { x: -1, y: -1 };
  let quietAt: { x: number; y: number } | null = null;
  function onPointerMove(event: PointerEvent): void {
    pointer = { x: event.clientX, y: event.clientY };
    if (quietAt && Math.hypot(pointer.x - quietAt.x, pointer.y - quietAt.y) > 4) quietAt = null;
  }
  function hush(): void {
    stopDwell();
    hovered = null;
    quietAt = { ...pointer };
  }
  function stopDwell(): void {
    if (dwell !== null) clearTimeout(dwell);
    dwell = null;
  }
  function show(subject: Subject | null, element: HTMLElement | null): void {
    if (dragging) return;
    stopDwell();
    if (!subject || !element) { hovered = null; return; }
    docked = subject;
    if (element.matches(':focus-visible')) { place(subject, element); return; }
    // Quiet is read when the dwell ends: the pointer may still be arriving.
    dwell = setTimeout(() => { dwell = null; if (!quietAt && element.isConnected) place(subject, element); }, DWELL_MS);
  }
  function place(subject: Subject, element: HTMLElement): void {
    const rect = element.getBoundingClientRect();
    const flip = rect.right + 290 > window.innerWidth - 290;
    hovered = { subject, x: flip ? rect.left - 282 : rect.right + 12, y: Math.max(48, Math.min(rect.top - 40, window.innerHeight - 440)) };
  }

  // The inspected object as it is in this frame, by id; a name alone when it has left.
  const permanents = $derived(new Map([...boardProjection.agent.battlefield, ...boardProjection.opponent.battlefield].map(permanent => [permanent.id, permanent])));
  const boardNames = $derived(new Map([...permanents].map(([id, permanent]) => [id, permanent.name ?? 'permanent'])));
  const cards = $derived(new Map(inspectableCards([boardProjection.agent, boardProjection.opponent]).map(card => [card.id, card])));
  const shown = $derived(inspectMode === 'dock' ? docked : hovered?.subject ?? null);
  const shownPermanent = $derived(shown?.permanent != null ? permanents.get(shown.permanent) ?? null : null);
  const shownCard = $derived(shown?.card != null ? cards.get(shown.card) ?? null : null);

  const playerName = (id: number): string => (id === boardProjection.agent.id ? you.name : opponent.name);
  const castTerms = $derived(permissionTerms(shownCard,
    id => permanents.get(id)?.name ?? `permanent #${id}`,
    playerName));
  const enteredUnder = $derived(
    shownPermanent?.entry_controller_id !== undefined
      ? possessive(playerName(shownPermanent.entry_controller_id))
      : null,
  );

  function onPermanent(item: PermanentItem, element: HTMLElement): void {
    if (onDraft(item.permanent.id)) return;
    if (joint) return;
    activate(item.permanent.name ?? 'Permanent', item.offers, element);
  }
  function previewPermanent(item: PermanentItem | null, element: HTMLElement | null): void {
    point(item?.offers ?? null);
    show(item ? { name: item.permanent.name ?? 'Permanent', text: item.permanent.text_box ?? null, permanent: item.permanent.id, card: null } : null, element);
  }
  function previewCard(card: CardState | null, element: HTMLElement | null): void {
    point(card ? board.hand.find(item => item.card.id === card.id)?.offers ?? null : null);
    show(card ? { name: card.name, text: card.text_box ?? null, permanent: null, card: card.id } : null, element);
  }
  /** A tray card inspects as itself: its rules text, counters and permissions. */
  function previewTray(item: TrayItem, element: HTMLElement | null): void {
    if (!item.card) { previewName(element ? item.name : null, element); return; }
    point(element ? [item.placed] : null);
    show(element ? { name: item.card.name, text: item.card.text_box ?? null, permanent: null, card: item.card.id } : null, element);
  }
  function previewName(name: string | null, element: HTMLElement | null): void {
    show(name ? { name, text: textOf(name), permanent: null, card: null } : null, element);
  }
  function onStack(item: StackItem, element: HTMLElement): void {
    activate(item.entry.label, item.offers, element);
  }

  /** Rules text for a name in the log, from any card of that name this viewer can see. */
  function textOf(name: string): string | null {
    const players = [boardProjection.agent, boardProjection.opponent];
    const card = inspectableCards(players).find(candidate => candidate.name === name);
    const permanent = players.flatMap(player => player.battlefield).find(candidate => candidate.name === name);
    return card?.text_box ?? permanent?.text_box ?? null;
  }
  function openZone(title: string, cards: CardState[], element: HTMLElement): void {
    const rect = element.getBoundingClientRect();
    zoneAt = zoneAt?.title === title ? null : { title, cards, x: rect.right + 8, y: Math.min(rect.top, window.innerHeight - 320) };
  }

  // ———————————————— Hand

  // Hand cards are large, 136 to 150 px wide (`HAND_MIN`, `HAND_MAX`): the
  // board gives the hand that height before its rows grow, and a crowded
  // board (24 permanents or more) may take it back down to `HAND_CROWDED`
  // when its rows need it (`handStrip`). A long hand's cards shrink before
  // they overlap (`handFan`), and the strip shrinks with them. An empty hand is a label.
  /** Strip heights, px: the card is the strip less 22 px of lift and padding, at 63:88. */
  const HAND_MIN = 212;
  const HAND_MAX = 232;
  const HAND_CROWDED = 160;
  /** The hand's most of the board's height: at 1280×800 it reaches `HAND_MAX` under the 64 px stop bar. */
  const HAND_SHARE = 0.35;
  const boardCount = $derived(boardProjection.agent.battlefield.length + boardProjection.opponent.battlefield.length);
  let boardHeight = $state(0);
  const handHeight = $derived(handStrip(boardHeight, [
    { creatures: pileShapes(board.opponent.creatures), support: pileShapes(board.opponent.support), width: theirRows.width },
    { creatures: pileShapes(board.you.creatures), support: pileShapes(board.you.support), width: yourRows.width },
  ], { gap: ROW_GAP, creatures: CREATURES, support: SUPPORT },
  { min: boardCount >= 24 ? HAND_CROWDED : HAND_MIN, max: Math.max(HAND_MIN, Math.min(HAND_MAX, Math.round(boardHeight * HAND_SHARE))), step: 8 }));
  /** The players' plates' column. */
  const PLATE_WIDTH = 148;
  /** The hand's cards keep clear of its label at the left edge. */
  const HAND_LABEL_WIDTH = 80;
  let handWidth = $state(0);
  /** A crowded hand's cards shrink to the width of a card in the crowded strip before they overlap. */
  const HAND_FLOOR = Math.floor((HAND_CROWDED - 22) * CARD_RATIO);
  const handFull = $derived(Math.max(56, Math.min(150, Math.floor((handHeight - 22) * CARD_RATIO))));
  // Cards offered from outside the board (a castable card in exile, Lessons outside the game) sit beside the hand, a size smaller, and take their width from it.
  const trayCard = $derived(Math.round(Math.min(handFull, 120) * 0.78));
  const trayReserve = $derived(view.tray.length && !finished && activeCursor === null ? Math.min(Math.floor(handWidth * 0.45), view.tray.length * (trayCard + 10) + 30) : 0);
  const fan = $derived(handFan(board.hand.length, handWidth - HAND_LABEL_WIDTH - 24 - trayReserve, { min: HAND_FLOOR, max: handFull }, 10));
  // Shrunk cards give the strip's spare height back to the board.
  const handTrack = $derived(board.hand.length === 0 && trayReserve === 0 ? '30px' : `${Math.min(handHeight, Math.ceil(fan.card / CARD_RATIO) + 22)}px`);

  // ———————————————— Pointer gestures

  const canMarquee = $derived(canDrag && drafting && preview?.kind === 'attackers' && activeCursor === null);
  const dropPermanents = $derived.by(() => {
    const hits = new Set<number>();
    if (snapped?.kind === 'object') hits.add(snapped.id);
    if (marquee && canMarquee && root) {
      for (const field of root.querySelectorAll('.rows [data-testid=field-row]')) {
        // Only scrolling fields clip their cards. Attackers in ordinary rows can lean past the edge.
        const area = field.classList.contains('overflow') ? overlap(marquee, field.getBoundingClientRect()) : marquee;
        if (!area) continue;
        for (const card of field.querySelectorAll<HTMLElement>('[data-draft][data-object-id]')) {
          const id = Number(card.dataset.objectId);
          if (declarable.has(id) && overlap(area, card.getBoundingClientRect())) hits.add(id);
        }
      }
    }
    return hits;
  });

  // Press a playable card in hand and drag it: it follows the pointer while
  // its slot steps back, and releasing it over the battlefield is a click on
  // it. Released anywhere else, or on Escape, it settles back with nothing sent.
  function pressHand(event: PointerEvent, item: HandItem): void {
    if (!canDrag || activeCursor !== null || joint || event.button !== 0 || !event.isPrimary || !playable(item.offers)) return;
    const opener = (event.currentTarget as HTMLElement).querySelector<HTMLElement>('[data-testid=board-card]');
    if (!opener) return;
    const rect = opener.getBoundingClientRect();
    gesture = press({ kind: 'play', overField: false, card: item.card, offers: item.offers, width: fan.card, grab: { x: event.clientX - rect.left, y: event.clientY - rect.top }, opener },
      event.pointerId, frame.revision, { x: event.clientX, y: event.clientY });
  }
  /** Resolve the element under the pointer against the same offers and preview used for clicks. */
  function connectTarget(drag: ConnectDrag, hit: Element | null): ConnectDrag {
    if (drag.blocker !== undefined) {
      const permanent = drafting ? hit?.closest<HTMLElement>('.rows [data-testid=board-card]') : null;
      const id = permanent ? Number(permanent.dataset.objectId) : null;
      const legal = id !== null && declarable.get(drag.blocker)?.can_block.some(attacker => attacker.entity === id);
      return { ...drag, target: legal ? { endpoint: { kind: 'object', id } } : null };
    }
    return { ...drag, target: offerTarget(hit) };
  }
  function offerTarget(hit: Element | null): OfferTarget | null {
    if (!targeting || !hit) return null;
    const permanent = hit.closest<HTMLElement>('.rows [data-testid=board-card]');
    if (permanent) {
      const id = Number(permanent.dataset.objectId);
      const item = fieldById.get(id);
      return item?.offers.length ? { endpoint: { kind: 'object', id }, title: item.permanent.name ?? 'Permanent', offers: item.offers, element: permanent } : null;
    }
    const stack = hit.closest<HTMLElement>('[data-stack-entry]');
    if (stack) {
      const id = Number(stack.dataset.stackEntry);
      const item = stackById.get(id);
      return item?.offers.length ? { endpoint: { kind: 'entry', id }, title: item.entry.label, offers: item.offers, element: stack } : null;
    }
    const plate = hit.closest<HTMLElement>('[data-testid=player-plate]');
    const element = plate?.querySelector<HTMLElement>('[data-testid=player-target]');
    if (!plate || !element) return null;
    const yours = plate.dataset.plateSide === 'you';
    const side = yours ? view.you : view.opponent;
    return side.offers.length ? { endpoint: { kind: 'player', index: side.player.player_index }, title: yours ? you.name : said, offers: side.offers, element } : null;
  }
  function onDragMove(event: PointerEvent): void {
    if (current.phase !== 'pressed' && current.phase !== 'dragging') return;
    const next = moveGesture(current, event.pointerId, { x: event.clientX, y: event.clientY }, drag => {
      if (drag.kind === 'marquee') return drag;
      const hit = document.elementFromPoint(event.clientX, event.clientY);
      return drag.kind === 'play' ? { ...drag, overField: !!hit?.closest('.board .half') } : connectTarget(drag, hit);
    });
    if (next === current) return;
    if (current.phase === 'pressed') {
      // Now a drag: inspect and open popovers close, and no text selection follows the pointer.
      hush();
      pointed = null;
      menuAt = null;
      zoneAt = null;
      window.getSelection()?.removeAllRanges();
    }
    gesture = next;
  }
  /** The carried card's box at `at`, as the hand draws it. */
  function carriedRect(at: { x: number; y: number }, width: number): DOMRect {
    return new DOMRect(at.x, at.y, width, footprint(width, true));
  }
  /** A carried card goes back to its slot in hand, from where it was let go. */
  function settleHome(drag: { opener: HTMLElement; width: number }, at: { x: number; y: number }): void {
    gesture = IDLE;
    if (drag.opener.isConnected) settle(drag.opener, carriedRect(at, drag.width), prefersReducedMotion());
  }
  function onDragEnd(event: PointerEvent): void {
    if (current.phase !== 'pressed' && current.phase !== 'dragging') return;
    if (event.pointerId !== current.pointerId) return;
    if (event.type === 'pointercancel') { endDrag(); return; }
    if (current.phase === 'dragging') onDragMove(event);
    const hits = dropPermanents;
    const ended = release(current, event.pointerId);
    const result = ended.release;
    gesture = ended.gesture;
    if (result.outcome === 'click' || result.outcome === 'none') return;
    // The click that follows this release is part of the drag.
    swallowClick = true;
    switch (result.outcome) {
      case 'cancel': settleHome(result.drag, result.at); return;
      case 'marquee':
        if (canMarquee && hits.size) { draft = addAttackers(draft, hits); stopNote = null; }
        return;
      case 'connect': {
        const { drag } = result;
        if (!drag.target) return;
        if (drag.blocker !== undefined) {
          draft = setBlock(draft, drag.blocker, drag.target.endpoint.id);
          armed = null;
          stopNote = null;
        } else activate(drag.target.title, drag.target.offers, drag.target.element);
        return;
      }
      case 'play':
        activate(result.drag.card.name, result.drag.offers, result.drag.opener, carriedRect(result.at, result.drag.width));
        return;
    }
  }
  /** Ends an unreleased gesture with nothing sent; a card already dropped stays held. */
  function endDrag(): void {
    if (current.phase === 'dragging' || current.phase === 'pressed') swallowClick = true;
    if (current.phase === 'dragging' && current.drag.kind === 'play') { settleHome(current.drag, carriedAt(current)!); return; }
    gesture = cancelGesture(gesture);
  }
  // Looking back or the board going read-only ends an unreleased drag. A hold
  // stays: committing a drop makes the board read-only until its answer.
  $effect(() => {
    if (!canDrag || activeCursor !== null) untrack(endDrag);
  });
  // A gesture from an earlier revision is gone: its frame was replaced.
  $effect(() => {
    const at = frame.revision;
    untrack(() => {
      if (gesture.phase === 'idle' || gesture.revision === at) return;
      if (gesture.phase !== 'holding') swallowClick = true;
      gesture = IDLE;
    });
  });
  // A held drop goes back to hand once what held it (a refused Command or its closed choices) ends at the same revision.
  $effect(() => {
    if (current.phase !== 'holding') return;
    const stillHeld = current.reason === 'pending' ? pending : openFor(current.drag.offers);
    if (!stillHeld) {
      const held = current;
      untrack(() => settleHome(held.drag, held.at));
    }
  });

  /** The open menu is the one for these offers (the menu's state is a copy, so compared by offer). */
  function openFor(offers: PlacedOffer[]): boolean {
    return menuAt !== null && menuAt.offers.length === offers.length && menuAt.offers.every((placed, index) => placed.offer.id === offers[index].offer.id);
  }

  // ———————————————— Keyboard

  // The board's keys, as the sheet `?` opens lists them. A recording has its
  // transport's keys instead of live play's.
  const recording = $derived(connection === null);
  const keys = $derived<{ key: string; does: string; plain?: boolean }[]>([
    ...(recording
      ? [
          { key: '← →', does: 'Step back or forward' },
          { key: 'Shift ← →', does: 'Previous or next decision' },
          { key: 'Home End', does: 'Start or end of the game' },
          { key: 'Space', does: 'Play or pause' },
          { key: 'Tab', does: 'Next region: fields, hand, stack, choices, moves' },
          { key: '↑ ↓', does: 'Move within a region' },
        ]
      : [
          { key: 'Space', does: 'Primary choice; return to now while looking back' },
          { key: '1–9', does: 'The numbered choice (returns to now and commits)' },
          { key: 'Enter', does: 'Act on the focused card or choice' },
          { key: 'Tab', does: 'Next region: fields, hand, stack, choices, history' },
          { key: '← →', does: 'Step history when no control has focus; move within a region' },
          { key: 'Shift Enter', does: 'Pass turn: pass every priority window until it ends (also F6)' },
          { key: 'Ctrl', does: 'Pause Auto-tap for the next payment (Full Control)' },
          { key: 'Shift Ctrl', does: 'Hold Full Control across turns' },
        ]),
    { key: 'Esc', does: recording ? 'Close a menu or preview' : 'Cancel a drag; close a menu or preview; return to now' },
    ...(canDrag || combatPreview !== undefined ? [{ key: 'Drag', does: 'Play a hand card, choose a target, assign a blocker, or box-select attackers', plain: true }] : []),
    { key: 'Inspect', does: 'Focus a card, or rest the pointer on it', plain: true },
    { key: '?', does: 'Show or hide these keys' },
  ]);
  let keysOpen = $state(false);
  let keysButton = $state<HTMLButtonElement | null>(null);
  const openingHandDecision = $derived(
    frame.projection.decision_context?.stage === 'opening_hand'
      || frame.offers.some(offer => offer.label === 'Keep hand' || offer.label === 'Mulligan'),
  );
  let myDeckOpen = $state(false);
  let oppDeckOpen = $state(false);
  let lastAutoTappedHash = '';
  let wasPayingInOnce = false;
  let ctrlPending = false;
  let ctrlShift = false;

  $effect(() => {
    if (recording || !live || activeCursor !== null || result || drafting || partitioning || partitionRunning || !onautotap) return;
    if (fullControl === 'once') {
      if (frame.action_space === 'PAY_OR_NOT') {
        wasPayingInOnce = true;
      } else if (wasPayingInOnce) {
        wasPayingInOnce = false;
        onfullcontrol?.('off');
      }
      return;
    }
    wasPayingInOnce = false;
    const candidate = autoTapOffer(frame, view, { autoTap, fullControl });
    if (!candidate || frame.frame_hash === lastAutoTappedHash) return;
    lastAutoTappedHash = frame.frame_hash;
    onautotap(candidate.offer, autoTapNote(frame));
  });
  let trackerFaces = $state.raw<Record<string, TrackerCardFace>>({});
  const trackerDefNames = $derived(boardProjection.definition_names ?? {});
  const selfTracker = $derived(buildSelfTracker(boardProjection.agent, trackerFaces, trackerDefNames));
  const oppTracker = $derived(buildOpponentTracker(boardProjection.opponent, trackerFaces, trackerDefNames));
  const anyTrackerOpen = $derived(myDeckOpen || oppDeckOpen);
  /** The reference column left of the board: the docked inspect pane, then the open deck trackers. */
  const sidebarOpen = $derived(anyTrackerOpen || inspectMode === 'dock');

  function toggleBothTrackers(): void {
    const next = !anyTrackerOpen;
    myDeckOpen = next && selfTracker !== null;
    oppDeckOpen = next;
  }

  $effect(() => {
    if (!myDeckOpen && !oppDeckOpen) return;
    const defNames = trackerDefNames;
    const resolveKeys = (counts: Record<string, number> | undefined): string[] =>
      Object.keys(counts ?? {}).map(key => defNames[key] ?? key);
    const names = [
      ...resolveKeys(boardProjection.agent.main_counts),
      ...resolveKeys(boardProjection.agent.sideboard_counts),
      ...resolveKeys(boardProjection.opponent.main_counts),
      ...resolveKeys(boardProjection.opponent.sideboard_counts),
      ...boardProjection.opponent.battlefield.map(p => p.name ?? ''),
      ...boardProjection.opponent.graveyard.map(c => c.name ?? ''),
      ...boardProjection.opponent.exile.map(c => c.name ?? ''),
      ...boardProjection.opponent.hand.map(c => c.name ?? ''),
    ].filter(Boolean);
    if (!names.length) return;
    void loadTrackerCardFaces(names).then(faces => { trackerFaces = faces; });
  });

  function closeKeys(): void {
    if (!keysOpen) return;
    keysOpen = false;
    keysButton?.focus();
  }

  function typing(target: EventTarget | null): boolean {
    return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
  }

  /** Space presses a focused control outside the board's regions; in them Enter acts and Space is the primary. */
  function ownsSpace(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    if (target.closest('[data-region]')) return false;
    return target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement || target.tagName === 'SUMMARY';
  }

  const TABBABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), summary, [tabindex]';
  /**
   * Tab past the last control comes back to the top bar, and Shift+Tab before
   * the first goes to the last, so a live game's Tab cycle is the top bar's
   * controls and the regions: never the browser's chrome or the page behind.
   */
  function wrapTab(event: KeyboardEvent): void {
    const stops = [...document.querySelectorAll<HTMLElement>(TABBABLE)]
      .filter(element => element.tabIndex >= 0 && !element.closest('[inert]') && element.getClientRects().length > 0);
    const edge = event.shiftKey ? stops[0] : stops.at(-1);
    if (!edge || document.activeElement !== edge) return;
    event.preventDefault();
    (event.shiftKey ? stops.at(-1) : stops[0])?.focus();
  }

  function onKeydown(event: KeyboardEvent): void {
    if (!recording && onfullcontrol && !typing(event.target)) {
      if (event.key === 'Control' && !event.metaKey && !event.altKey) {
        if (!event.repeat) {
          ctrlPending = true;
          ctrlShift = event.shiftKey;
        }
        return;
      }
      if (event.key === 'Shift' && ctrlPending) {
        ctrlShift = true;
        return;
      }
      if (ctrlPending) {
        // Another key was pressed with Ctrl (e.g. Ctrl+C / Ctrl+R): not bare Ctrl.
        ctrlPending = false;
      }
    }
    if (event.key === 'Tab' && !recording && !event.metaKey && !event.ctrlKey && !event.altKey) { wrapTab(event); return; }
    if (event.metaKey || event.ctrlKey || event.altKey || typing(event.target)) return;
    // Escape during a drag ends the drag and nothing else.
    if (event.key === 'Escape' && (current.phase === 'pressed' || current.phase === 'dragging')) {
      event.preventDefault();
      endDrag();
      return;
    }
    if (event.key === '?') { event.preventDefault(); keysOpen = !keysOpen; return; }
    if (event.key === 'Escape' && keysOpen) { closeKeys(); return; }
    if (event.key === 'Escape' && anyTrackerOpen) { myDeckOpen = false; oppDeckOpen = false; return; }
    if (event.key === 'Escape' && running) { stopSending(); return; }
    if (event.key === 'Escape' && partitionRunning) { stopPartitionSending(); return; }
    if (event.key === 'Escape' && live && armed !== null) { armed = null; return; }
    if (event.key === 'Escape' && (menuAt || zoneAt || hovered)) { closeMenu(); zoneAt = null; hovered = null; stopDwell(); return; }
    if (event.key === 'Escape' && activeCursor !== null) { clearActiveCursor(); return; }
    if (!recording && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      const target = event.target as HTMLElement | null;
      const inControl = target !== null && target !== document.body && target !== document.documentElement
        && !target.closest('.log, .walk-bar')
        && (target.closest('button, a, input, select, textarea, summary, [data-testid=stop-bar], [data-region]') !== null);
      if (!inControl) {
        event.preventDefault();
        setCursor(stepCursor(history, cursor, event.key === 'ArrowLeft' ? -1 : 1), 'line');
        return;
      }
    }
    if (event.key === 'F6' || (event.key === 'Enter' && event.shiftKey)) { event.preventDefault(); passTurn(); return; }
    if (/^[1-9]$/.test(event.key)) {
      // A joint combat or pile-partition draft owns the whole decision; its folded per-item offers are not numbered shortcuts.
      if (!menuAt && (joint || partitioning || partitionRunning)) return;
      const placed = (menuAt?.offers ?? view.list).find(candidate => candidate.number === Number(event.key)) ?? null;
      if (placed && live) { event.preventDefault(); act(placed); }
      return;
    }
    if (event.key === ' ') {
      // Space while looking back (or walking) only returns to now and never acts.
      if (!recording && cursor !== null) { event.preventDefault(); returnToNow(); return; }
      // An open card menu takes Space as a press of its focused item.
      if (menuAt) return;
      // Never let Space re-activate a previously clicked history line or scrubber tick.
      const inHistoryOrWalk = event.target instanceof HTMLElement && event.target.closest('.log, .walk-bar') !== null;
      if (inHistoryOrWalk) event.preventDefault();
      if (!inHistoryOrWalk && ownsSpace(event.target)) return;
      if (drafting) { event.preventDefault(); confirmDraft(); return; }
      if (partitioning) { event.preventDefault(); confirmPartition(); return; }
      if (view.primary && live) { event.preventDefault(); act(view.primary); }
    }
  }

  function onKeyup(event: KeyboardEvent): void {
    if (event.key === 'Control' && ctrlPending && !recording && onfullcontrol) {
      const shift = ctrlShift || event.shiftKey;
      ctrlPending = false;
      ctrlShift = false;
      onfullcontrol(nextFullControl(fullControl, shift));
    }
  }
</script>

<svelte:window onkeydown={onKeydown} onkeyup={onKeyup} onblur={() => { ctrlPending = false; ctrlShift = false; endDrag(); }} onpointermove={event => { onPointerMove(event); onDragMove(event); }} onpointerup={onDragEnd} onpointercancel={onDragEnd}
  onpointerdown={event => {
    if (keysOpen && !(event.target instanceof Element && event.target.closest('.keys, .keys-hint'))) keysOpen = false;
    if (sheet?.open && !(event.target instanceof Node && sheet.contains(event.target))) sheet.open = false;
  }} />

<div class="game" class:dragging class:with-sidebar={sidebarOpen} data-testid="game" data-action-space={frame.action_space} data-live={live} data-thinking={thinking !== null}
  data-turn={turnNumber} data-step={turnStep} data-turn-owner={yourTurn ? 'you' : 'opponent'} style:--plate={`${PLATE_WIDTH}px`} style:--hand-label={`${HAND_LABEL_WIDTH}px`} bind:this={root}>
  <header class="topbar chrome-bar">
    <a href="/games" class="brand" aria-label="Academy · Games">Academy</a>
    <p class="matchup type-caption" data-testid="deck-names">
      <span class="text-ink">{you.name}</span>{#if you.deck}<span class="text-ink-3">{' · '}{you.deck}</span>{/if}
      <span class="text-ink-3">{' vs '}</span>
      <span class="text-ink" title={[opponent.name, opponent.description].filter(Boolean).join(' — ')}>{said}</span>{#if opponent.deck}<span class="text-ink-3">{' · '}{opponent.deck}</span>{/if}
    </p>
    {#if activeCursorReading}
      <p class="turn cursor-reading" data-testid="turn-number" data-cursor={cursor} title={activeCursorReading}>
        <span class="cursor-text" data-testid="cursor-reading">{activeCursorReading}</span><span class="cursor-sep" aria-hidden="true">&nbsp;·&nbsp;</span><button type="button" class="esc-now" data-testid="cursor-now" onclick={clearActiveCursor}>Esc for now</button>
      </p>
    {:else}
      <p class="turn">{#if turnStep !== null && boardProjection.turn.creature_died_this_turn}A creature died this turn{/if}</p>
    {/if}
    <div class="right">
      {#if connection}
        <!-- Connected says nothing: only a problem shows; the status stays for assistive technology. -->
        <span data-testid="connection-badge" data-connection-state={connection} role="status" aria-live="polite" aria-atomic="true"
          aria-label={`Connection status: ${connectionLabel}`} class="connection type-label" class:sr-only={connection === 'connected'}>{connection === 'connected' ? connectionLabel : connectionLabel.charAt(0).toUpperCase() + connectionLabel.slice(1)}</span>
      {/if}
      <button type="button" class="keys-hint type-caption" data-testid="keys-hint" aria-haspopup="dialog" aria-expanded={keysOpen}
        aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)" bind:this={keysButton} onclick={() => (keysOpen = !keysOpen)}><kbd>?</kbd></button>
      <!-- The bar says what is on; the controls are in Settings. A setting off its default is a chip here, and pressing it turns it back. -->
      {#if !recording && ontoggleautotap && (fullControl !== 'off' || !autoTap)}
        <button type="button" class="chip type-caption" data-testid="chip-auto-tap" title={fullControl === 'hold' ? 'Full Control (hold) · Shift+Ctrl or click to resume Auto-tap' : fullControl === 'once' ? 'Full Control (next payment) · Ctrl or click to resume Auto-tap' : 'Auto-tap is off · click to turn it on'}
          onclick={ontoggleautotap}>{fullControl === 'hold' ? 'Full control · hold' : fullControl === 'once' ? 'Full control' : 'Manual tap'}<span class="x" aria-hidden="true">×</span></button>
      {/if}
      {#each chips as chip (chip.key)}
        <button type="button" class="chip type-caption" data-testid={`chip-${chip.key}`} title="Click to turn off" onclick={chip.clear}>{chip.label}<span class="x" aria-hidden="true">×</span></button>
      {/each}
      {#if anyTrackerOpen}<button type="button" class="chip type-caption" data-testid="chip-decks" title="Click to hide the deck lists" onclick={toggleBothTrackers}>Decks<span class="x" aria-hidden="true">×</span></button>{/if}
      {#if inspectMode === 'dock'}<button type="button" class="chip type-caption" data-testid="chip-preview" title="Click to show card previews beside the card again" onclick={toggleInspect}>Preview pinned<span class="x" aria-hidden="true">×</span></button>{/if}
      {#if bar}{@render bar()}{/if}
      <a class="bar-link type-caption" href={historyHref}>History</a>
      <details class="game-menu" bind:this={sheet}>
        <summary class="btn btn-ghost btn-sm" data-testid="game-menu">Settings</summary>
        <div class="menu-panel" role="region" aria-label="Settings">
          {#if (!recording && ontoggleautotap) || menu?.play}
            <section class="group" aria-label="Play">
              <p class="type-label text-ink-3">Play</p>
              {#if !recording && ontoggleautotap}
                <label class="setting"><span>Auto-tap<small>{fullControl === 'hold' ? 'Full control is held · Shift+Ctrl resumes' : fullControl === 'once' ? 'Full control for the next payment · Ctrl resumes' : 'Ctrl pauses once · Shift+Ctrl holds full control'}</small></span>
                  <input type="checkbox" role="switch" data-testid="auto-tap-toggle" data-full-control={fullControl} checked={fullControl === 'off' && autoTap} onchange={ontoggleautotap} /></label>
              {/if}
              {#if menu?.play}{@render menu.play()}{/if}
            </section>
          {/if}
          {#if menu?.analysis}<section class="group" aria-label="Analysis"><p class="type-label text-ink-3">Analysis</p>{@render menu.analysis()}</section>{/if}
          <section class="group" aria-label="View">
            <p class="type-label text-ink-3">View</p>
            <label class="setting"><span>Deck lists<small>What is left in each deck, in the left sidebar</small></span>
              <input type="checkbox" role="switch" data-testid="deck-tracker-toggle" checked={anyTrackerOpen} onchange={toggleBothTrackers} /></label>
            <label class="setting"><span>Pin the card preview<small>In the left sidebar, instead of beside the card you point at</small></span>
              <input type="checkbox" role="switch" data-testid="inspect-toggle" checked={inspectMode === 'dock'} onchange={toggleInspect} /></label>
            <div class="setting"><span>Theme</span><ThemeSwitch /></div>
          </section>
          {#if menu?.game}<section class="group" aria-label="Game"><p class="type-label text-ink-3">Game</p>{@render menu.game()}</section>{/if}
        </div>
      </details>
    </div>
  </header>

  {#if sidebarOpen}
    <aside class="sidebar" aria-label="Inspected card and deck trackers" data-testid="reference-sidebar">
      {#if inspectMode === 'dock'}{@render dock()}{/if}
      {#if oppDeckOpen}
        <DeckTracker title={oppTracker.mode === 'opponent-open' ? (opponent.deck ?? `${said}'s deck`) : `${said} · seen`} view={oppTracker} faces={trackerFaces} side="opponent" onclose={() => { oppDeckOpen = false; }} />
      {/if}
      {#if myDeckOpen && selfTracker}
        <DeckTracker title={you.deck ?? 'Your deck'} view={selfTracker} faces={trackerFaces} side="you" onclose={() => { myDeckOpen = false; }} />
      {/if}
    </aside>
  {/if}

  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions (clicking an earlier board returns to now; Esc and Space do the same from the keyboard) -->
  <div class="board" id="board" data-testid="game-board" aria-label="Board" bind:this={boardEl} bind:clientHeight={boardHeight} style:grid-template-rows={`minmax(0, 1fr) minmax(0, 1fr) ${handTrack}`}
    data-board={activeCursor !== null ? (cursorHit ? 'step' : 'missing') : 'frame'}
    onclick={event => { if (activeCursor !== null && !(event.target instanceof Element && event.target.closest('.zone, .combat-table-sheet'))) clearActiveCursor(); }}>
    <section use:region={{ across: connection !== null }} class="half opponent" aria-label={`${possessive(said)} battlefield`}>
      <Plate player={boardProjection.opponent} definitionNames={boardProjection.definition_names} drop={snapped?.kind === 'player' && snapped.index === boardProjection.opponent.player_index} name={said} caption={[decidesBy(opponent.name), opponent.deck].filter(Boolean).join(' · ') || null} offers={board.opponent.offers} {live} damage={damage.players.get(boardProjection.opponent.player_index) ?? 0}
        stopLifeDelta={stopDeltas.players.get(boardProjection.opponent.player_index) ?? 0}
        choosing={opponentChoosing} pointed={pointedPlayerIndex === boardProjection.opponent.player_index} side="opponent" onactivate={element => activate(said, board.opponent.offers, element)} onzone={(zone, cards, element) => openZone(`${said} · ${zone}`, cards, element)}
        ondeck={() => { oppDeckOpen = !oppDeckOpen; }} deckOpen={oppDeckOpen} />
      <div class="rows" class:selectable={canMarquee} bind:clientWidth={theirRows.width} bind:clientHeight={theirRows.height} style:grid-template-rows={rowsStyle(theirSplit, theirRows, false)}>
        <div class="row support"><Field piles={board.opponent.support} label={`${possessive(said)} lands and other permanents`} limits={SUPPORT} {live} asked={boardAsked} draft={draftMarks} sources={fieldSources} {connectTargets} drop={dropPermanents} {touched} beat={activeSubjects} damage={damage.objects} stopDeltas={stopDeltas.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} toward="down" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row creatures"><Field piles={board.opponent.creatures} label={`${possessive(said)} creatures`} limits={CREATURES} {live} asked={boardAsked} draft={draftMarks} sources={fieldSources} {connectTargets} drop={dropPermanents} {touched} beat={activeSubjects} damage={damage.objects} stopDeltas={stopDeltas.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} deaths={opponentDeaths} toward="down" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section use:region={{ across: connection !== null }} class="half you" aria-label="Your battlefield">
      <Plate player={boardProjection.agent} drop={snapped?.kind === 'player' && snapped.index === boardProjection.agent.player_index} name={you.name} caption={you.deck} offers={board.you.offers} {live} damage={damage.players.get(boardProjection.agent.player_index) ?? 0}
        stopLifeDelta={stopDeltas.players.get(boardProjection.agent.player_index) ?? 0}
        pointed={pointedPlayerIndex === boardProjection.agent.player_index}
        side="you" onactivate={element => activate(you.name, board.you.offers, element)} onzone={(zone, cards, element) => openZone(`Your ${zone.toLowerCase()}`, cards, element)}
        ondeck={selfTracker ? () => { myDeckOpen = !myDeckOpen; } : undefined} deckOpen={myDeckOpen} />
      <div class="rows" class:selectable={canMarquee} bind:clientWidth={yourRows.width} bind:clientHeight={yourRows.height} style:grid-template-rows={rowsStyle(yourSplit, yourRows, true)}>
        <div class="row creatures"><Field piles={board.you.creatures} label="Your creatures" limits={CREATURES} {live} asked={boardAsked} draft={draftMarks} sources={fieldSources} {connectTargets} drop={dropPermanents} {touched} beat={activeSubjects} damage={damage.objects} stopDeltas={stopDeltas.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} deaths={yourDeaths} {acting} toward="up" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row support"><Field piles={board.you.support} label="Your lands and other permanents" limits={SUPPORT} {live} asked={boardAsked} draft={draftMarks} sources={fieldSources} {connectTargets} drop={dropPermanents} {touched} beat={activeSubjects} damage={damage.objects} stopDeltas={stopDeltas.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} {acting} toward="up" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section use:region={{ across: connection !== null }} class="hand" aria-label={`Your hand, ${board.hand.length} cards`} bind:clientWidth={handWidth}>
      <span class="type-heading hand-label text-ink-2">Hand · {board.hand.length}</span>
      <div class="hand-cards" style:width={`${fan.card + Math.max(0, board.hand.length - 1) * fan.step}px`}>
        {#each board.hand as item, index (item.card.id)}
          {@const away = carried?.card.id === item.card.id}
          <!-- svelte-ignore a11y_no_static_element_interactions (the press starts a pointer drag; the card's button is the control) -->
          <div class="hand-card" class:playable={live && item.offers.length > 0} class:draggable={canDrag && playable(item.offers)} style:left={`${index * fan.step}px`}
            onpointerdown={event => pressHand(event, item)}>
            <BoardCard name={item.card.name} width={fan.card} slice={fan.covered && index < board.hand.length - 1 ? fan.step : null} card={item.card} offers={item.offers} {live} asked={board.asked.has(item.card.id)}
              dim={away || (acting && !openingHandDecision && item.offers.length === 0) ? 'hand' : null} travel={!away}
              onactivate={element => activate(item.card.name, item.offers, element)}
              onpreview={element => previewCard(element ? item.card : null, element)} />
          </div>
        {/each}
      </div>
      {#if view.tray.length && !finished && !partitioning && !partitionRunning && activeCursor === null}
        <div class="tray" role="group" aria-label="Choices from outside the board" data-testid="tray" style:max-width={`${trayReserve}px`}>
          {#each view.tray as item (item.placed.offer.id)}
            {#if item.name}
              <div class="tray-card" style:width={`${trayCard}px`}>
                <span class="type-caption tray-caption" title={offerText(item.placed.offer)}>{offerText(item.placed.offer)}</span>
                <BoardCard name={item.name} width={trayCard} offers={[item.placed]} {live}
                  onactivate={() => act(item.placed)}
                  onpreview={element => previewTray(item, element)} />
              </div>
            {:else}
              <button type="button" class="btn btn-secondary btn-sm" disabled={!live} onclick={() => act(item.placed)}>
                <kbd>{item.placed.number}</kbd>{offerText(item.placed.offer)}
              </button>
            {/if}
          {/each}
        </div>
      {/if}
    </section>

    {#if (partitioning || partitionRunning) && partition && !finished}
      <!-- Over the board, foldable so the battlefield stays readable around it. -->
      <section use:region={{ across: connection !== null }} class="partition-panel" class:folded={partitionFolded} aria-label={`${partition.sourceName}: split into two piles`} data-testid="partition-panel">
        {#if partitionFolded}
          <button type="button" class="btn btn-secondary btn-sm" data-testid="partition-unfold" onclick={() => (partitionFolded = false)}>Show pile split</button>
        {:else}
          <header class="partition-head">
            <div>
              <p class="type-heading text-ink">{partition.sourceName} · Split into two piles</p>
              <p class="type-caption text-ink-2">Click a card to move it between piles. Your opponent sees the face-up pile and only the count of face-down cards.</p>
            </div>
            <button type="button" class="fold type-caption" data-testid="partition-fold" onclick={() => (partitionFolded = true)}>Look at the board</button>
          </header>
          <div class="partition-piles">
            {#each [
              {
                key: 'face_up',
                testId: 'partition-pile-up',
                title: 'Face-up',
                hint: 'Opponent sees these',
                cards: partitionUpCards,
                draft: 'in' as const,
                note: 'in the face-up pile; press to move to the face-down pile',
                empty: 'Empty — click a card below to put it face-up',
              },
              {
                key: 'face_down',
                testId: 'partition-pile-down',
                title: 'Face-down',
                hint: 'Hidden — opponent sees count only',
                cards: partitionDownCards,
                draft: 'can' as const,
                note: 'in the face-down pile; press to move to the face-up pile',
                empty: 'Empty — all remaining cards are face-up',
              },
            ] as pile (pile.key)}
              <div class="partition-pile" data-testid={pile.testId} role="group" aria-label={`${pile.title} pile, ${pile.cards.length} ${pile.cards.length === 1 ? 'card' : 'cards'}`}>
                <p class="partition-pile-head">
                  <span class="type-label text-ink">{pile.title} · {pile.cards.length}</span>
                  <span class="type-caption text-ink-3">{pile.hint}</span>
                </p>
                {#if pile.cards.length}
                  <div class="partition-cards">
                    {#each pile.cards as card (card.id)}
                      <div class="partition-card-slot" data-testid="partition-card" data-object-id={card.id} data-pile={pile.key}>
                        <BoardCard name={card.name} width={104} {card} live={partitioning} draft={pile.draft}
                          note={pile.note}
                          onactivate={() => togglePartitionCard(card.id)}
                          onpreview={element => previewCard(element ? card : null, element)} />
                      </div>
                    {/each}
                  </div>
                {:else}
                  <p class="partition-empty type-caption text-ink-3">{pile.empty}</p>
                {/if}
              </div>
            {/each}
          </div>
          {#if partition.committed.length}
            <p class="partition-committed type-caption text-ink-3" data-testid="partition-committed">
              Already placed ({partition.committed.length}): {partition.committed.map(card => card.name).join(', ')}
            </p>
          {/if}
        {/if}
      </section>
    {/if}

    {#if dragging && carried}
      <!-- Where releasing plays the card, and what it does: the battlefield above the hand. -->
      <div class="drop-zone" class:over={carried.overField} style:bottom={handTrack} data-testid="drop-zone" data-over={carried.overField} aria-hidden="true">
        <span class="drop-label type-caption">{dropLabel(carried)}</span>
      </div>
    {/if}

    {#if finished}
      <!-- Over the board, not over the screen: the final position stays readable around it. -->
      <section class="result-card" class:folded={resultFolded} aria-label="Game result" data-testid="result-card">
        {#if resultFolded}
          <button type="button" class="btn btn-secondary btn-sm" onclick={() => (resultFolded = false)}>Show the result</button>
        {:else}
          {@render result?.()}
          <button type="button" class="fold type-caption" onclick={() => (resultFolded = true)}>Look at the board</button>
        {/if}
      </section>
    {/if}
    {#if deathMarks.some(mark => mark.animate)}
      <div class="death-layer" aria-hidden="true">
        {#each deathMarks as mark (mark.key)}
          {#if mark.animate}
            <div class="death-travel" data-testid="death-travel" data-death-object={mark.entity} data-death-rule={mark.rule ?? ''}
              data-death-side={mark.side}
              style:left={`${mark.x}px`} style:top={`${mark.y}px`} style:width={`${mark.width}px`} style:height={`${mark.height}px`}
              use:runDeathTravel={mark}>
              <span class="death-travel-name">{mark.name}</span>
              {#if mark.rule}<span class="death-travel-rule">{mark.rule}</span>{/if}
            </div>
          {/if}
        {/each}
      </div>
    {/if}
    {#if combatTable}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions (resting on the combat table holds the walk and raises a row's combatants) -->
      <section
        class="combat-table-sheet"
        aria-label={combatTable.heading}
        data-testid="combat-table"
        data-creatures={combatTable.creatureCount}
        onmouseenter={() => (combatTableHovered = true)}
        onmouseleave={() => { combatTableHovered = false; combatRowSubjects = []; }}
      >
        <header class="combat-table-head">
          <span class="type-heading text-ink">{combatTable.heading}</span>
          {#if walking && combatTableHovered}<span class="type-caption text-ink-3">holding</span>{/if}
        </header>
        <div class="combat-table-scroll">
          <table>
            <thead>
              <tr><th>Attacker</th><th>Blocked by</th><th>Dealt</th><th>Took</th><th>Result</th></tr>
            </thead>
            <tbody>
              {#each combatTable.rows as row (row.attacker.entity)}
                <tr
                  data-testid="combat-table-row"
                  onmouseenter={() => (combatRowSubjects = row.subjects)}
                  onmouseleave={() => (combatRowSubjects = [])}
                >
                  <td><b>{row.attacker.name}</b>{#if row.attacker.pt}<span class="pt-note">{' '}{row.attacker.pt}</span>{/if}</td>
                  <td>{#if row.blockers.length}{row.blockers.map(b => `${b.name}${b.pt ? ` ${b.pt}` : ''}`).join(', ')}{:else}<span class="text-ink-3">unblocked</span>{/if}</td>
                  <td>{row.dealt}</td>
                  <td class="num">{row.took}</td>
                  <td>{row.result}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}
  </div>

  <div class="rail-slot" id="rail-choices">
    <Rail {view} stack={board.stack} {frame} {live} {line} {hint} {primaryText} {status} {whose} {thinking} {opponentChoosing} {workStarted} {interrupted} {canPassTurn} {recentFacts} resolvedSpells={recent.resolved} {history}
      opponentName={said} onact={act} onpassturn={passTurn} {paymentNote} onpointoffer={placed => point(placed ? [placed] : null)}
      onstack={onStack} {stackSources} drop={snapped?.kind === 'entry' ? snapped.id : undefined} onpreviewname={previewName} {notices} over={finished}
      {evidence} {moveMarks} {shownMoves} {recapTitle} {onjump} recording={readOnly} {currentMove}
      onselectline={connection !== null ? selectHistoryLine : undefined}
      cursorLineIds={active.lineIds} cursorSourceSeq={active.sourceSeq}
      draft={joint ? combatControls : partitioning || partitionRunning ? partitionControls : undefined} stopped={stopNote} />
  </div>

  {#snippet partitionControls()}
    <div class="combat-draft" data-testid="partition-draft">
      {#if partitionRunning}
        <button type="button" class="btn btn-secondary" data-testid="partition-stop" aria-disabled={activeCursor !== null || undefined} onclick={stopPartitionSending}><kbd>Esc</kbd>Stop sending</button>
      {:else if partition && partitionSummary}
        <button type="button" class="btn btn-primary primary" data-testid="primary-action" data-draft-ready="true"
          aria-label={`Confirm: ${partitionSummary.plan}`} title={partitionSummary.plan}
          aria-disabled={activeCursor !== null || undefined} onclick={confirmPartition}>
          <span>{partitionSummary.text}</span><kbd>Space</kbd>
        </button>
        <button type="button" class="btn btn-secondary secondary" data-testid="partition-swap" aria-disabled={activeCursor !== null || undefined}
          onclick={() => { stopNote = null; partitionFaceUp = swapPiles(partition.remaining, partitionFaceUp); }}>Swap piles</button>
        <p class="type-caption text-ink-3">Nothing is placed until you confirm. Click any card to move it between piles.</p>
      {/if}
    </div>
  {/snippet}

  {#snippet combatControls()}
    <div class="combat-draft" data-testid="combat-draft" data-kind={preview?.kind}>
      {#if running}
        <button type="button" class="btn btn-secondary" data-testid="combat-stop" onclick={stopSending}><kbd>Esc</kbd>Stop sending</button>
      {:else if words}
        <button type="button" class="btn btn-primary primary" data-testid="primary-action" data-draft-ready={exact !== null && !exact.refused}
          aria-label={`Confirm: ${exact?.refused ? exact.refused.reason : words.plan}`} title={words.plan}
          disabled={!live || !exact || exact.refused !== null} onclick={confirmDraft}>
          <span>{words.text}</span><kbd>Space</kbd>
        </button>
        {#each counts as group (group.ids[0])}
          <div class="pile-count" data-testid="pile-count" role="group" aria-label={`${group.name}: ${group.drafted} of ${group.ids.length} attacking`}>
            <span class="type-caption"><b>{group.drafted}</b> of {group.ids.length} {group.name}</span>
            <button type="button" class="step" aria-label={`One fewer ${group.name}`} disabled={!live || group.drafted === 0}
              onclick={() => { draft = setPileCount(draft, group.ids, group.drafted - 1); }}>−</button>
            <button type="button" class="step" aria-label={`One more ${group.name}`} disabled={!live || group.drafted === group.ids.length}
              onclick={() => { draft = setPileCount(draft, group.ids, group.drafted + 1); }}>+</button>
          </div>
        {/each}
        {#if draft.attackers.length || draft.blocks.length}
          <button type="button" class="btn btn-secondary secondary" data-testid="combat-clear" disabled={!live} onclick={() => { draft = EMPTY_DRAFT; armed = null; }}>Clear the draft</button>
        {/if}
        <p class="type-caption text-ink-3">Nothing is declared until you confirm. {preview?.kind === 'blockers' ? BLOCK_INSTRUCTION : ATTACK_INSTRUCTION}.</p>
      {/if}
    </div>
  {/snippet}

  {#snippet stripToggle()}
    <!-- Live play's strip only counts, so Tab passes it by; a recording's opens to a navigable line. -->
    <button type="button" class="strip-toggle" aria-expanded={strip!.open} onclick={strip!.ontoggle} data-testid="strip-toggle"
      tabindex={strip!.onjump ? undefined : -1}>
      <span aria-hidden="true">{strip!.open ? '▾' : '▸'}</span> Decisions · {strip!.marks.filter(mark => mark.decision).length}
    </button>
  {/snippet}
  {#if showWalkBar || bottom || strip?.open}
    <div class="bottom">
      {#if showWalkBar}
        <div class="walk-bar" data-testid="walk-bar" data-walking={walking} data-cursor={cursor ?? undefined}>
          <div class="walk-row">
            <!-- svelte-ignore a11y_no_static_element_interactions (resting the pointer on the caption holds the walk) -->
            <div class="walk-caption" data-testid="walk-caption" data-walking={walking} data-holding={walking && captionHovered}
              aria-live="polite" title={active.caption || undefined}
              onmouseenter={() => (captionHovered = true)} onmouseleave={() => (captionHovered = false)}>
              {#if cursor !== null}
                <span class="walk-text">{active.caption}</span>
                {#if boardNotice}
                  <span class="walk-note" data-testid={boardNotice.testid}> · {boardNotice.text}</span>
                {/if}
              {:else}
                <span class="walk-idle">Turn {scrubber.turn ?? turnNumber} · {scrubber.stops.length} {scrubber.stops.length === 1 ? 'stop' : 'stops'}</span>
              {/if}
            </div>
            <div class="walk-actions type-caption">
              {#if cursor !== null}
                {#if activeStopIndex >= 0}
                  <span class="walk-count" data-testid="walk-count">{activeStopIndex + 1} of {scrubber.stops.length}{#if walking && (captionHovered || combatTableHovered)}{' · holding'}{/if}</span>
                {/if}
                <button type="button" class="walk-btn" data-testid="walk-now" onclick={returnToNow}>Now <kbd>Esc</kbd></button>
              {/if}
              {#if scrubber.stops.length > 0}
                <button type="button" class="walk-btn" data-testid="walk-replay" onclick={replayTurn}>Replay turn</button>
              {/if}
              {#if onwalkpace}
                <span class="walk-pace" role="group" aria-label="Move pace" data-testid="walk-pace">
                  <button type="button" class="pace-btn" class:on={walkPace === 'study'} aria-pressed={walkPace === 'study'} data-pace="study" title="1.4 seconds per stop" onclick={() => onwalkpace('study')}>1.4s</button>
                  <button type="button" class="pace-btn" class:on={walkPace === 'quick'} aria-pressed={walkPace === 'quick'} data-pace="quick" title="0.7 seconds per stop" onclick={() => onwalkpace('quick')}>Quick</button>
                  <button type="button" class="pace-btn" class:on={walkPace === 'off'} aria-pressed={walkPace === 'off'} data-pace="off" title="Do not auto-walk stops" onclick={() => onwalkpace('off')}>Off</button>
                </span>
              {/if}
            </div>
          </div>
          <div class="walk-scrubber" data-testid="walk-scrubber" role="group" aria-label={`Turn ${scrubber.turn ?? turnNumber} stops`}>
            {#each scrubber.stops as stop, idx (stop.seq)}
              {@const isActive = idx === activeStopIndex}
              <button type="button" class="walk-tick" class:move={stop.moveStart} class:yours={stop.yoursOnly} class:active={isActive}
                class:passed={activeStopIndex >= 0 && idx < activeStopIndex}
                data-testid="walk-tick" data-stop-seq={stop.seq} data-move-start={stop.moveStart ? 'true' : undefined}
                aria-pressed={isActive}
                aria-label={`Stop ${idx + 1} of ${scrubber.stops.length}: ${stop.caption}`}
                title={stop.caption}
                onclick={() => setCursor(stop.seq, 'stop')}>
                <span class="tick-bar" aria-hidden="true"></span>
              </button>
            {/each}
          </div>
        </div>
      {/if}
      {@render bottom?.()}
      {#if strip?.open}<Strip marks={strip.marks} selected={strip.selected} onjump={strip.onjump} note={strip.note} scale={strip.scale} />{/if}
    </div>
  {/if}
  <nav class="stops-slot" aria-label={`Turn ${turnNumber}, ${yourTurn ? 'yours' : "opponent's"}: steps and stops`}>
    <StopBar {track} turn={turnNumber} cursorActive={Boolean(activeCursorReading)} oncycle={oncyclestop} aside={strip ? stripToggle : undefined} />
  </nav>
  <Lines lines={standingLines} {draftLine} {root} revision={`${frame.revision}:${cursor ?? 'now'}:${pointed?.offer.id ?? 'none'}`} />
  {#if marquee}
    <div class="marquee" aria-hidden="true" data-testid="attack-marquee" style:left={`${marquee.left}px`} style:top={`${marquee.top}px`}
      style:width={`${marquee.right - marquee.left}px`} style:height={`${marquee.bottom - marquee.top}px`}></div>
  {/if}
  {#if carried && carriedPoint}
    <!-- The carried card: a copy for the eye, following the pointer; the card in hand stays the control. -->
    <div class="carried" inert aria-hidden="true" data-testid="drag-ghost" data-holding={current.phase === 'holding' ? current.reason : undefined}
      style:translate={`${carriedPoint.x}px ${carriedPoint.y}px`}>
      <BoardCard name={carried.card.name} width={carried.width} card={carried.card} />
    </div>
  {/if}
</div>

{#if menuAt}
  <div class="menu" use:region role="menu" tabindex="-1" aria-label={`Choices for ${menuAt.title}`} bind:this={menuElement} data-testid="card-menu"
    style:left={`${menuAt.x}px`} style:top={`${menuAt.y}px`}>
    <p class="type-heading text-ink-2">{menuAt.title}</p>
    {#each menuAt.offers as placed (placed.offer.id)}
      <button type="button" role="menuitem" onclick={() => act(placed)} data-offer-id={placed.offer.id}><kbd>{placed.number}</kbd>{offerText(placed.offer)}</button>
    {/each}
    <button type="button" role="menuitem" class="cancel" onclick={closeMenu}><kbd>Esc</kbd>Cancel</button>
  </div>
{/if}

{#if keysOpen}
  <div class="keys" role="dialog" aria-label="Keyboard shortcuts" data-testid="keys-sheet">
    <p class="type-heading text-ink-2">Keys{recording ? ' · Review' : ''}</p>
    <dl>
      {#each keys as row (row.key)}
        <dt>{#if row.plain}<span class="type-caption">{row.key}</span>{:else}<kbd>{row.key}</kbd>{/if}</dt><dd class="type-caption">{row.does}</dd>
      {/each}
    </dl>
    <button type="button" class="cancel type-caption" onclick={closeKeys}><kbd>Esc</kbd>Close</button>
  </div>
{/if}

{#if zoneAt}
  <div class="menu zone" role="dialog" aria-label={zoneAt.title} style:left={`${zoneAt.x}px`} style:top={`${zoneAt.y}px`}>
    <p class="type-heading text-ink-2">{zoneAt.title}</p>
    <ol>
      {#each zoneAt.cards as card (card.id)}
        <li><button type="button" onmouseenter={event => previewCard(card, event.currentTarget)} onmouseleave={() => previewCard(null, null)}
          onfocus={event => previewCard(card, event.currentTarget)} onblur={() => previewCard(null, null)}>{card.name}</button></li>
      {/each}
    </ol>
    <button type="button" class="cancel" onclick={() => (zoneAt = null)}><kbd>Esc</kbd>Close</button>
  </div>
{/if}

{#snippet dock()}
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (a scrolling region must be reachable by keyboard) -->
  <section class="dock" aria-label="Inspect" data-testid="inspect-dock" tabindex="0">
    {#if docked}
      <Inspect name={docked.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? docked.text} permanent={shownPermanent} card={shownCard} {castTerms} {enteredUnder}
        blocks={shownPermanent?.blocking !== undefined ? boardNames.get(shownPermanent.blocking) ?? null : null} art={72} />
    {:else}
      <p class="type-body text-ink-3">Hover or focus a card to inspect it.</p>
    {/if}
  </section>
{/snippet}

{#if inspectMode === 'float' && hovered}
  <div class="preview" aria-hidden="true" style:left={`${hovered.x}px`} style:top={`${hovered.y}px`} data-testid="card-preview">
    <Inspect name={hovered.subject.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? hovered.subject.text} permanent={shownPermanent} card={shownCard} {castTerms} {enteredUnder}
      blocks={shownPermanent?.blocking !== undefined ? boardNames.get(shownPermanent.blocking) ?? null : null} />
  </div>
{/if}

<style>
  /*
   * The board's one layering order, lowest first: cards, the lines between
   * cards and their targets, floating panels (the result card),
   * then what a drag draws (the carried card, the drop zone and the marquee).
   * Each row of permanents and the hand keep their own stacking, so a raised,
   * attacking or lifted card orders only among its neighbours and never climbs
   * over a line or a panel. Above the whole game (itself at 10) float the inspect
   * preview (250), then card menus and the keys sheet (300). Add no other z-index.
   */
  .game {
    --layer-lines: 1; --layer-panel: 2; --layer-drag: 3;
    position: fixed; inset: 0; z-index: 10; display: grid;
    grid-template-columns: minmax(0, 1fr) clamp(250px, 21vw, 300px);
    grid-template-rows: 40px minmax(0, 1fr) auto auto;
    background: var(--bg); color: var(--text);
  }
  /* Reference on the left (the inspected card, the decks), the board in the middle, the game's present on the right. */
  .game.with-sidebar {
    grid-template-columns: 256px minmax(0, 1fr) clamp(250px, 21vw, 300px);
  }
  .sidebar {
    grid-row: 2 / span 2;
    grid-column: 1;
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    background: var(--bg-surface);
  }
  /* The inspect pane holds two fifths of the column whatever card it shows, so the trackers under it never move as you hover; they share the rest and scroll inside. */
  .sidebar > :global(*) { flex: 1 1 0; min-height: 0; }
  .sidebar > :global(* + *) { border-top: 1px solid var(--border-strong); }
  .sidebar > .dock { flex: none; height: 40%; padding: 10px 14px; }
  .topbar {
    grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 0 14px;
    border-bottom: 1px solid var(--border); background: var(--bg-surface); min-width: 0;
  }
  .brand { font-family: var(--font-serif); font-weight: 600; font-size: 20px; line-height: 28px; color: var(--accent-text); flex-shrink: 0; }
  .matchup { min-width: 0; flex: 0 2 auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* The matchup gives way first: the turn and the controls keep their width; a long cursor reading truncates cleanly before Esc for now. */
  .turn { margin: 0 auto; flex-shrink: 0; font-size: 14px; line-height: 20px; font-weight: 600; white-space: nowrap; }
  .turn.cursor-reading {
    display: flex; align-items: baseline; min-width: 0; flex: 0 1 auto; overflow: hidden;
    color: var(--accent-text);
  }
  .cursor-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cursor-sep, .esc-now { flex: none; }
  .esc-now { font: inherit; min-height: 0; color: var(--text-secondary); text-decoration: underline dotted; cursor: pointer; }
  .esc-now:hover { color: var(--text); }
  .stops-slot { grid-row: 4; grid-column: 1 / -1; display: grid; grid-template-columns: minmax(0, 1fr) clamp(250px, 21vw, 300px); }
  .right { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .connection { color: var(--warning-text); }

  .board {
    grid-row: 2; grid-column: 1;
    position: relative; min-height: 0; min-width: 0; display: grid;
    transition: grid-template-rows var(--motion-move) var(--motion-move-ease);
  }
  .game.with-sidebar .board { grid-column: 2; }
  /* One row the half's own height: a tall plate scrolls inside it rather than growing the rows past the half. */
  .half { display: grid; grid-template-columns: var(--plate) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); min-height: 0; border-bottom: 1px solid var(--border); background: var(--bg-muted); }
  .half :global(.plate) { border-right: 1px solid var(--border); }
  .rows { isolation: isolate; display: grid; min-height: 0; min-width: 0; padding: 6px 10px; gap: 6px; }
  .opponent .rows { grid-template-rows: minmax(0, 42fr) minmax(0, 58fr); }
  .you .rows { grid-template-rows: minmax(0, 58fr) minmax(0, 42fr); }
  .row { min-height: 0; min-width: 0; }
  .row.support { border-top: 1px dashed transparent; }
  .opponent .row.creatures, .you .row.support { padding-top: 4px; }

  .hand { position: relative; isolation: isolate; min-height: 0; display: flex; align-items: flex-end; justify-content: center; padding: 0 12px 8px var(--hand-label); overflow: hidden; }
  .hand-label { position: absolute; left: 14px; bottom: 10px; }
  .hand-cards { position: relative; height: 100%; }
  .hand-card { position: absolute; bottom: 0; transition: transform var(--motion-fast) ease-out; }
  .hand-card:hover, .hand-card:focus-within { transform: translateY(-10px); z-index: 30; }
  /* A playable card is dragged, never panned; while a card is carried, its neighbours stay still. */
  .rows.selectable, .hand-card.draggable { touch-action: none; }
  .game :global([data-connect-source]) { touch-action: none; }
  .game :global(button[data-connect-target]:not([data-drop])) { outline: 2px dashed var(--accent); outline-offset: 2px; }
  .game :global(button[data-drop]) { outline: 2px solid var(--accent); outline-offset: 2px; }
  .dragging { user-select: none; }
  .dragging .hand-card:hover, .dragging .hand-card:focus-within { transform: none; }

  .marquee {
    position: fixed; z-index: var(--layer-drag); pointer-events: none;
    border: 1px dashed var(--accent); background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  .carried { position: fixed; left: 0; top: 0; z-index: var(--layer-drag); pointer-events: none; }
  .carried :global(.tcard) { box-shadow: var(--lift-selected); }
  .drop-zone {
    position: absolute; left: 0; right: 0; top: 0; z-index: var(--layer-drag); pointer-events: none;
    display: flex; justify-content: center; align-items: flex-start; padding-top: 12px;
    border: 2px dashed var(--border-strong); border-radius: var(--radius-md);
  }
  .drop-zone.over { border-color: var(--accent); }
  .drop-label {
    padding: 2px 10px; border-radius: var(--radius-full); border: 1px solid var(--border-strong);
    background: var(--bg-surface); color: var(--text-secondary); box-shadow: var(--shadow-raised);
  }
  .drop-zone.over .drop-label { border-color: var(--accent); color: var(--text); }

  /* Beside the hand, never over the battlefield: the same strip holds what you can play from your hand and from elsewhere. */
  .tray { flex: none; align-self: stretch; display: flex; align-items: flex-end; gap: 10px; margin-left: 14px; padding-left: 14px; border-left: 1px solid var(--border-strong); overflow-x: auto; }
  .tray-card { flex: none; display: grid; justify-items: center; gap: 4px; text-align: center; }
  .tray-caption { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; color: var(--text-secondary); }

  .result-card {
    position: absolute; left: 50%; top: 44%; transform: translate(-50%, -50%); z-index: var(--layer-panel); width: min(440px, 80%);
    display: grid; gap: 10px; padding: 18px 20px 14px; border: 1px solid var(--border-strong); border-radius: var(--radius-md);
    background: var(--bg-surface); box-shadow: var(--shadow-raised);
    animation: appear var(--motion-move) ease-out;
  }
  .combat-table-sheet {
    position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%); z-index: var(--layer-panel);
    width: min(640px, calc(100% - 28px)); max-height: 46%; display: flex; flex-direction: column;
    border: 1px solid var(--border-strong); border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--bg-surface) 96%, transparent); box-shadow: var(--shadow-raised);
    overflow: hidden;
  }
  .combat-table-head {
    display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
    padding: 6px 12px; border-bottom: 1px solid var(--border); background: var(--bg-field);
  }
  .combat-table-scroll { overflow-y: auto; padding: 4px 12px 8px; }
  .combat-table-sheet table { width: 100%; border-collapse: collapse; font: 14px/20px var(--font-sans); }
  .combat-table-sheet th {
    text-align: left; font: 600 12px/16px var(--font-sans);
    color: var(--text-tertiary); padding: 4px 8px 4px 0; border-bottom: 1px solid var(--border);
  }
  .combat-table-sheet td { padding: 4px 8px 4px 0; border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent); vertical-align: top; color: var(--text); }
  .combat-table-sheet td b { font-family: var(--font-serif); font-weight: 600; }
  .combat-table-sheet td.num { font-variant-numeric: tabular-nums; }
  .combat-table-sheet .pt-note { font-variant-numeric: tabular-nums; color: var(--text-secondary); }
  .combat-table-sheet tbody tr:hover td { background: var(--bg-field); }
  .combat-table-sheet tbody tr:last-child td { border-bottom: 0; }
  @keyframes appear { from { opacity: 0; } }
  /* Steady-height caption + scrubber bar under the board once a move has occurred. */
  .walk-bar {
    height: 54px; display: grid; grid-template-rows: 22px 22px; align-content: center; gap: 2px;
    padding: 4px 14px;
  }
  .walk-row { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .walk-caption {
    flex: 1 1 auto; min-width: 0; display: flex; align-items: baseline; white-space: nowrap; overflow: hidden;
    font: 14px/20px var(--font-serif); color: var(--text);
  }
  .walk-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .walk-note, .walk-idle { flex: none; font: 12px/16px var(--font-sans); color: var(--text-secondary); }
  .walk-actions { flex: none; display: inline-flex; align-items: center; gap: 8px; color: var(--text-secondary); }
  .walk-count { font-variant-numeric: tabular-nums; color: var(--text-tertiary); }
  .walk-btn {
    min-height: 0; display: inline-flex; align-items: center; gap: 4px; padding: 1px 6px; border-radius: var(--radius-sm);
    color: var(--text-secondary); white-space: nowrap;
  }
  .walk-btn:hover, .walk-btn:focus-visible { color: var(--text); background: var(--bg-field); }
  .walk-pace { display: inline-flex; align-items: center; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
  .pace-btn {
    min-height: 0; padding: 1px 6px; font: 500 12px/16px var(--font-sans); color: var(--text-secondary); background: none; border: 0; cursor: pointer;
  }
  .pace-btn + .pace-btn { border-left: 1px solid var(--border); }
  .pace-btn:hover, .pace-btn:focus-visible { color: var(--text); background: var(--bg-field); }
  .pace-btn.on { background: var(--text); color: var(--bg-surface); }
  .walk-scrubber { display: flex; align-items: center; gap: 3px; height: 22px; min-width: 0; }
  .walk-tick {
    flex: 1 1 0; min-width: 8px; max-width: 56px; height: 20px; display: flex; align-items: center;
    padding: 0; border: 0; background: none; cursor: pointer;
  }
  .walk-tick .tick-bar {
    width: 100%; height: 3px; border-radius: var(--radius-sm); background: var(--border-strong);
    transition: background-color var(--motion-fast) ease-out, height var(--motion-fast) ease-out;
  }
  .walk-tick.move .tick-bar { height: 6px; }
  .walk-tick.yours .tick-bar { background: color-mix(in srgb, var(--info-text) 50%, var(--border-strong)); }
  .walk-tick.passed .tick-bar { background: color-mix(in srgb, var(--accent) 55%, var(--border-strong)); }
  .walk-tick.active .tick-bar { background: var(--accent); height: 8px; }
  .walk-tick:hover .tick-bar, .walk-tick:focus-visible .tick-bar { background: var(--text); }
  .result-card.folded { top: auto; bottom: 12px; left: auto; right: 12px; transform: none; width: auto; padding: 6px; }
  .result-card .fold { justify-self: end; color: var(--text-secondary); }
  .result-card .fold:hover { color: var(--text); }

  .partition-panel {
    position: absolute; left: 50%; top: 44%; transform: translate(-50%, -50%); z-index: var(--layer-panel);
    width: min(680px, 92%); max-height: 82%; overflow-y: auto;
    display: grid; gap: 12px; padding: 16px 18px 14px;
    border: 1px solid var(--border-strong); border-radius: var(--radius-md);
    background: var(--bg-surface); box-shadow: var(--shadow-raised);
    animation: appear var(--motion-fast) ease-out;
  }
  .partition-panel.folded { top: auto; bottom: 12px; left: auto; right: 12px; transform: none; width: auto; max-height: none; padding: 6px; }
  .partition-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
  .partition-head .fold { flex: none; color: var(--text-secondary); }
  .partition-head .fold:hover { color: var(--text); }
  .partition-piles { display: grid; gap: 10px; }
  .partition-pile {
    display: grid; gap: 8px; padding: 10px 12px;
    border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg-field);
  }
  .partition-pile-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .partition-cards { display: flex; flex-wrap: wrap; gap: 8px; }
  .partition-empty { padding: 10px 0; }
  .partition-committed { border-top: 1px solid var(--border); padding-top: 8px; }

  .combat-draft { display: grid; gap: 6px; }
  /* The draft's buttons are the rail's primary and secondary, drawn by the board. */
  .combat-draft .primary { height: 40px; padding-block: 6px; width: 100%; font-size: 14px; justify-content: center; white-space: nowrap; }
  .combat-draft .primary span { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .combat-draft .primary kbd { flex: none; opacity: 1; }
  .combat-draft .primary:disabled { background: var(--bg-muted); color: var(--text-secondary); opacity: 1; }
  .combat-draft .secondary { width: 100%; min-height: 32px; padding-block: 5px; justify-content: flex-start; }
  .pile-count { display: flex; align-items: center; gap: 6px; }
  .pile-count span { flex: 1; }
  .pile-count .step { width: 24px; height: 22px; border: 1px solid var(--border-strong); border-radius: var(--radius-sm); font-weight: 600; }
  .pile-count .step:disabled { opacity: 0.4; }
  .rail-slot { min-height: 0; grid-row: 2 / span 2; grid-column: 2; }
  .game.with-sidebar .rail-slot { grid-column: 3; }
  .strip-toggle { font: 14px/20px var(--font-sans); color: var(--text-secondary); white-space: nowrap; }
  .strip-toggle:hover { color: var(--text); }
  .bottom { grid-row: 3; grid-column: 1; min-width: 0; border-top: 1px solid var(--border); background: var(--bg-surface); }
  .game.with-sidebar .bottom { grid-column: 2; }

  .menu {
    position: fixed; z-index: 300; min-width: 220px; max-width: 320px; padding: 8px; display: grid; gap: 2px;
    border: 1px solid var(--border-strong); border-radius: var(--radius-md); background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .menu, .preview { animation: open var(--motion-fast) ease-out; transform-origin: top left; }
  @keyframes open { from { opacity: 0; transform: scale(0.97); } }
  .menu p { padding: 0 6px 4px; }
  .menu button { display: flex; align-items: center; gap: 8px; text-align: left; padding: 6px 8px; border-radius: var(--radius-sm); font-size: 14px; line-height: 20px; color: var(--text); width: 100%; }
  .menu button:hover, .menu button:focus-visible { background: var(--bg-field); outline: 1px solid var(--border-strong); }
  .menu .cancel { color: var(--text-secondary); font-size: 12px; line-height: 16px; }
  .zone ol { max-height: 260px; overflow-y: auto; }

  .preview {
    position: fixed; z-index: 250; width: 270px; max-height: calc(100vh - 60px); overflow: hidden; pointer-events: none;
    padding: 10px; border-radius: var(--radius-md); border: 1px solid var(--border-strong); background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .dock { overflow-y: auto; }
  .keys-hint { display: inline-flex; align-items: center; color: var(--text-tertiary); padding: 2px 3px; border-radius: var(--radius-sm); }
  .keys-hint:hover, .keys-hint[aria-expanded='true'] { color: var(--text); background: var(--bg-field); }
  .keys {
    position: fixed; z-index: 300; top: 46px; right: 14px; width: 340px; padding: 10px 12px 8px; display: grid; gap: 6px;
    border: 1px solid var(--border-strong); border-radius: var(--radius-md); background: var(--bg-surface); box-shadow: var(--shadow-raised);
    animation: open var(--motion-fast) ease-out; transform-origin: top right;
  }
  .keys dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 12px; align-items: baseline; }
  .keys dt { text-align: right; white-space: nowrap; }
  .keys dd { color: var(--text-secondary); }
  .keys .cancel { justify-self: end; display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); padding: 3px 6px; border-radius: var(--radius-sm); }
  .keys .cancel:hover, .keys .cancel:focus-visible { background: var(--bg-field); }
  .bar-link { color: var(--text-secondary); padding: 2px 6px; border-radius: var(--radius-sm); }
  .bar-link:hover { color: var(--text); background: var(--bg-field); }
  .chip { padding: 1px 9px; border-radius: var(--radius-md); white-space: nowrap; color: var(--text); background: var(--bg-field); border: 1px solid var(--border-strong); }
  .chip .x { margin-left: 6px; color: var(--text-tertiary); }
  .chip:hover { background: var(--bg-muted, var(--bg-field)); border-color: var(--text-tertiary); }
  /* The Settings sheet: every control of the table, grouped; it scrolls inside the window rather than running off it. */
  .game-menu { position: relative; }
  .game-menu > summary { list-style: none; }
  .game-menu > summary::-webkit-details-marker { display: none; }
  .menu-panel {
    position: absolute; right: 0; top: calc(100% + 8px); z-index: 320; width: 360px; max-width: calc(100vw - 28px); max-height: calc(100dvh - 64px); overflow-y: auto;
    display: grid; gap: 14px; padding: 14px; border: 1px solid var(--border-strong); border-radius: var(--radius-md);
    background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .group { display: grid; gap: 8px; justify-items: start; }
  .group + .group { padding-top: 12px; border-top: 1px solid var(--border); }
  .group > :global(*) { width: 100%; }
  .group > :global(a.btn) { width: auto; }
  .setting, .group :global(.setting) { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; line-height: 20px; cursor: pointer; }
  .setting small, .group :global(.setting small) { display: block; font-size: 12px; line-height: 16px; color: var(--text-tertiary); }
  .setting input, .group :global(.setting input) { flex: none; accent-color: var(--accent); }

  .death-layer { position: absolute; inset: 0; pointer-events: none; z-index: var(--layer-panel); overflow: hidden; }
  .death-travel {
    position: absolute; display: flex; flex-direction: column; justify-content: space-between; padding: 4px;
    border-radius: var(--radius-card); border: 1px solid var(--damage);
    background: var(--bg-surface); box-shadow: var(--shadow-raised); color: var(--text);
  }
  .death-travel-name { font: 600 12px/16px var(--font-serif); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .death-travel-rule {
    align-self: flex-start; padding: 0 5px; border-radius: var(--radius-sm);
    font: 600 12px/16px var(--font-sans); color: var(--bg); background: var(--damage);
  }

  @media (prefers-reduced-motion: reduce) {
    .hand-card, .board { transition: none; }
    .menu, .preview, .result-card, .partition-panel, .keys { animation: none; }
  }
</style>
