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
   */
  import { onMount, setContext, untrack, type Snippet } from 'svelte';
  import ThemeSwitch from '$lib/components/ThemeSwitch.svelte';
  import type { PreviewState } from '$lib/payment-preview';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import type { CardState, ConnectionState, ExperienceFrame, GameLogEntry, InteractionOffer, StopsConfig, StopSide } from '$lib/types';
  import Field from './Field.svelte';
  import Inspect from './Inspect.svelte';
  import { permissionTerms } from './inspect';
  import Lines from './Lines.svelte';
  import { prefersReducedMotion, Travel } from './motion';
  import { region } from './region';
  import { BEAT_MS, beatsFor, turnAt, type Beat } from './beats';
  import { boardAt, type StepBoard } from './boards';
  import { moveList, type MoveTurn } from './moves';
  import { EVIDENCE, nameSeats, type OfferEvidence, type StripLine } from './replay';
  import { paymentPlan, previewable } from './payment';
  import { addressOf, committedOf, drawDraft, draftWords, EMPTY_DRAFT, isDeclaration, loadDraft, pileCounts as draftPiles, planKey, planOf, saveDraft,
    setPileCount, stepOffer, toggleAttacker, toggleBlock, type CombatFetcher, type CombatPreview, type DeclarationStep, type Draft } from './combat';
  import { offerText, primaryLabel, promptHint, promptLine } from './prompt';
  import { nameBook, recentView, boardLines, turnOwners, visibleNames, type RecentView } from './recent';
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
  import { CARD_RATIO, handFan, handStrip, inspectableCards, pileShapes, possessive, splitHalf, boardView, type HalfSplit, type PermanentItem, type PlacedOffer, type RowLimits, type StackItem, type TrayItem } from './view';

  interface Props {
    frame: ExperienceFrame;
    live: boolean;
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
    log: GameLogEntry[];
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
    onpassturn?: () => void;
    /** Where "since your last move" starts, when the caller knows (a recording); else the person's last commit. */
    since?: number | null;
    /** The move list, when the caller keeps it (a recording); else read from `log`. */
    moves?: MoveTurn[];
    /** Jump to the position a move names; moves are buttons when set. */
    onjump?: (id: string) => void;
    /** Nothing here can be answered (a recording, or a policy's line through it): the rail drops its answer buttons and empty stack. Defaults to `onjump` being set. */
    readOnly?: boolean;
    /** The move the shown position belongs to. */
    currentMove?: string | null;
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
    menu?: Snippet;
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
  }

  let {
    frame, live, status, you, opponent, stops, onceStops, oncyclestop, connection = null, connectionLabel = '', log, canPassTurn,
    presentation, thinking, opponentChoosing, workStarted, interrupted, payment = null, onpreview = () => {}, onact, onpassturn = () => {},
    since: sinceGiven, moves: movesGiven, onjump, readOnly = onjump !== undefined, currentMove = null, evidence = null, moveMarks, shownMoves, recapTitle, heading = null, bottom, strip, notices, result, menu,
    combatPreview, stepBoards = [],
    autoTap = false, fullControl = 'off', ontoggleautotap, onfullcontrol, onautotap,
  }: Props = $props();

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
  let running = $state.raw<{ kind: CombatPreview['kind']; steps: DeclarationStep[]; at: number; sentAt: number } | null>(null);
  let stopNote = $state<string | null>(null);
  let draftFrame = '';
  const declaring = $derived(live && combatPreview !== undefined && isDeclaration(frame));
  const preview = $derived(previewed && previewed.frameHash === frame.frame_hash ? previewed.preview : null);
  const drafting = $derived(declaring && !running && preview !== null && preview.declarable.length > 0);
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
  /** Each creature's place in the draft and what pressing it does. */
  const draftMarks = $derived.by(() => {
    if (!drafting || !preview) return null;
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
        places.set(id, armed === id ? 'armed' : blocking !== undefined ? 'in' : 'can');
        notes.set(id, armed === id ? 'placing this block; press an attacker it can block'
          : blocking !== undefined ? `drafted to block ${names.get(blocking) ?? 'an attacker'}; press to move or take back the block`
            : creature.must_block ? 'must block; press, then the attacker it blocks' : 'can block; press, then the attacker it blocks');
      }
      const placing = armed === null ? null : declarable.get(armed);
      for (const attacker of placing?.can_block ?? []) {
        const id = attacker.entity;
        const taken = draft.blocks.some(([b, a]) => b === armed && a === id);
        places.set(id, 'can');
        notes.set(id, `${taken ? 'take back the block by' : 'block it with'} ${names.get(armed!) ?? 'the blocker'}`);
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
    if (!drafting || !preview) return false;
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
    if (!ready || ready.refused || !ready.steps.length) return;
    const offer = stepOffer(frame, ready.kind, ready.steps[0]);
    if (!offer) { stopNote = 'The board changed before the plan was sent; draft again.'; return; }
    running = { kind: ready.kind, steps: ready.steps, at: 0, sentAt: frame.revision };
    send(offer);
  }

  // Each frame after a sent step must be the decision the preview predicted for the next.
  $effect(() => {
    const revision = frame.revision;
    untrack(() => {
      const sending = running;
      if (!sending || revision === sending.sentAt) return;
      const next = sending.at + 1;
      if (next >= sending.steps.length) { running = null; draft = EMPTY_DRAFT; return; }
      const offer = live ? stepOffer(frame, sending.kind, sending.steps[next]) : null;
      if (!offer) {
        running = null;
        draft = EMPTY_DRAFT;
        stopNote = `Stopped after ${next} of ${sending.steps.length}: the game asked something the preview did not predict. The board shows where it stands; draft again from here.`;
        return;
      }
      running = { ...sending, at: next, sentAt: revision };
      onact(offer);
    });
  });
  function stopSending(): void {
    if (!running) return;
    stopNote = `Stopped after ${running.at + 1} of ${running.steps.length}; the rest was not sent.`;
    running = null;
    draft = EMPTY_DRAFT;
  }

  /** The frame as the board draws it: with the drafted declaration, while there is one. */
  const drafted = $derived(drafting || running ? drawDraft(frame, draft) : frame);
  const view = $derived(boardView(drafted, players));
  const whose = $derived(heading ?? (frame.status === 'game_over' ? 'Game over'
    : !frame.prompt ? 'Waiting'
      : frame.prompt.actor === projection.agent.player_index
        ? you.name === 'You' ? (live ? 'Your decision' : 'Your decision · waiting') : `${possessive(you.name)} decision`
        : `${possessive(said)} decision`));
  // ———————————————— Payment preview

  // The offer the board previews: the one a person points at or focuses, when
  // the authority can preview it (a cast, a target, a tap), else the pending
  // payment itself. Previewing is a read-only query; it commits nothing.
  let pointed = $state<PlacedOffer | null>(null);
  const payOffer = $derived(view.primary?.offer.verb === 'pay' ? view.primary : null);
  // Pointing at the pending payment's own row is still that payment: the
  // prompt keeps its plan and note, so nothing shifts under the pointer.
  const previewing = $derived(!live ? null
    : pointed && previewable(pointed.offer) && pointed.offer.id !== payOffer?.offer.id ? pointed : payOffer);
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
    : running ? `Declaring ${running.at + 1} of ${running.steps.length}…`
      : promptLine(frame, view, onjump === undefined));
  // How to answer with the live input; a recording has none.
  const hint = $derived(drafting || running || onjump !== undefined ? null : promptHint(frame, view));
  const paymentNote = $derived(!payOffer ? null
    : payment?.error && previewing === payOffer ? 'The payment preview is unavailable; Pay still pays automatically.'
      : payKnown && payKnown.state !== 'exact' && payKnown.state !== 'not_applicable' ? payKnown.explanation : null);
  function point(offers: PlacedOffer[] | null): void {
    pointed = offers?.length === 1 ? offers[0] : null;
  }

  // ———————————————— What happened since your last move

  // The revision your last move was made at; events after it are "since your last move".
  let since = $state.raw<{ revision: number; frame: ExperienceFrame; logLength: number } | null>(null);
  const sinceRevision = $derived(sinceGiven !== undefined ? sinceGiven : since?.revision ?? null);
  const book = $derived(nameBook(since && sinceGiven === undefined ? [since.frame, frame] : [frame], since && sinceGiven === undefined ? log.slice(since.logLength) : []));
  const recent = $derived(recentView(frame, sinceRevision, presentation.events, index => presentation.labelsAt(index), players, book));

  // ———————————————— The opponent's moves, one beat at a time

  // Live play only (a recording has its transport). Facts already on screen
  // when the board opens are history, not news.
  let shownSeq = untrack(() => presentation.events.at(-1)?.seq ?? -1);
  let shownMatch = untrack(() => frame.match_id);
  let shownRevision = untrack(() => frame.revision);
  let wasConnected = untrack(() => connection === 'connected');
  let beats = $state.raw<Beat[]>([]);
  let beatAt = $state(0);
  const beat = $derived(beats[beatAt] ?? null);
  const beatSubjects = $derived(new Set(beat?.subjects ?? []));
  // Before the DOM updates: a frame that brings beats draws the first beat's board first, never the final one.
  $effect.pre(() => {
    const facts = recent.facts;
    const lastSeq = presentation.events.at(-1)?.seq ?? -1;
    const { match_id, revision } = frame;
    const connected = connection === 'connected';
    if (connection === null) return;
    untrack(() => {
      if (!connected || !wasConnected || match_id !== shownMatch || revision < shownRevision) {
        endReadOut();
        shownSeq = lastSeq;
      } else {
        const fresh = prefersReducedMotion() ? [] : beatsFor(facts, shownSeq, players);
        if (fresh.length || revision !== shownRevision) { beats = fresh; beatAt = 0; }
        shownSeq = Math.max(shownSeq, lastSeq);
      }
      shownMatch = match_id;
      shownRevision = revision;
      wasConnected = connected;
    });
  });
  $effect(() => {
    if (!beat) return;
    const timer = setTimeout(() => { beatAt += 1; }, BEAT_MS);
    return () => clearTimeout(timer);
  });
  /**
   * The read-out holds the whole game surface: while a beat plays, the first
   * press, click or acting key anywhere in the game ends it and does nothing
   * else, so the live decision answers only once the board shows its own
   * position. Focus keys (Tab, arrows) still move.
   */
  const readingOut = $derived(beat !== null);
  function endReadOut(): void { beats = []; beatAt = 0; }
  // Damage marks and the plate's badge add up the whole window since your last
  // move; a beat's board may be from before that damage, so they wait with the
  // decision. So does the result card: the game ends once its last beat is read.
  const NO_DAMAGE: RecentView['damage'] = { objects: new Map(), players: new Map() };
  const damage = $derived(readingOut ? NO_DAMAGE : recent.damage);
  const finished = $derived(result !== undefined && !readingOut);

  // Each beat is shown on the board it happened on, when the authority sent
  // it (`stepBoards`); else on the frame's. A past board has no offers, and the
  // rail's choices wait until the read-out ends.
  const beatBoard = $derived(beat ? boardAt(stepBoards, beat.seq) : null);
  const drawn = $derived(beatBoard ? { ...frame, projection: beatBoard, offers: [] } : drafted);
  const board = $derived(beatBoard ? boardView(drawn, players) : view);
  const boardProjection = $derived(drawn.projection);
  const boardNames = $derived(beatBoard ? new Map([...beatBoard.agent.battlefield, ...beatBoard.opponent.battlefield]
    .map(permanent => [permanent.id, permanent.name ?? 'permanent'])) : names);
  const owners = $derived(turnOwners(frame, presentation.events));
  // The turn the top bar and stop bar name: the beat's own while one plays. A
  // beat's board may be a later turn's (one engine step can run from an attack
  // into the next turn), and then no step of the beat's turn is current.
  const turnNumber = $derived(beat ? turnAt(presentation.events, beat.seq) ?? boardProjection.turn.turn_number : boardProjection.turn.turn_number);
  const turnStep = $derived(turnNumber === boardProjection.turn.turn_number ? boardProjection.turn.step : null);
  const yourTurn = $derived(turnStep !== null ? boardProjection.turn.active_player_id === boardProjection.agent.id : owners.get(turnNumber) === 'you');
  const track = $derived(phaseTrack(turnStep, stops, yourTurn, onceStops));
  const lines = $derived(boardLines(drawn, presentation.events, recent.resolved, sinceRevision));
  /** Permanents something on the stack targets: a dashed edge in the damage colour. */
  const targeted = $derived(new Set(lines.flatMap(line => line.kind === 'target' && line.to.kind === 'object' ? [line.to.id] : [])));
  /** The person may act on the drawn board now: what they cannot act on steps back. */
  const acting = $derived(live && !readingOut && frame.prompt?.actor === projection.agent.player_index && (view.list.length > 0 || drafting));
  /** The pointer press that ended a read-out; its click acts on nothing. */
  let endedByPress = false;
  function onPress(event: PointerEvent): void {
    if (event.button !== 0) return;
    endedByPress = readingOut;
    if (endedByPress) {
      event.stopPropagation();
      endReadOut();
    }
  }
  function onClick(event: MouseEvent): void {
    const endedByPointer = event.detail > 0 && endedByPress;
    if (event.detail > 0) endedByPress = false;
    if (!readingOut && !endedByPointer) return;
    event.preventDefault();
    event.stopPropagation();
    endReadOut();
  }
  // Capturing listeners on the whole game: board, rail, tray and menus end a read-out alike.
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
  // Both players' moves are narrated as Review narrates them, players named by seat.
  const seatName = (seat: 0 | 1): string => (seat === projection.agent.player_index ? 'You' : said);
  const named = $derived(log.map(entry => (entry.actor === 'villain' || entry.actor === 'hero' ? { ...entry, text: nameSeats(entry.text, seatName, entry.actor === 'hero') } : entry)));
  const moves = $derived(movesGiven ?? moveList(named, players, owners, visibleNames(frame, presentation.labels)));

  // ———————————————— Motion

  let root = $state<HTMLElement | null>(null);
  const travel = new Travel();
  // Cards travel between frames and between the boards beats are shown on.
  $effect.pre(() => {
    void boardProjection;
    travel.snapshot(root);
  });
  $effect(() => {
    void boardProjection;
    travel.play(root, prefersReducedMotion());
  });

  // Battlefield frames are printed cards, upright or tapped on their side;
  // lands and other support permanents are smaller cards of the same frame.
  const CREATURES: RowLimits = { gap: 8, minCard: 48, maxCard: 140, maxLines: 3 };
  const SUPPORT: RowLimits = { gap: 6, minCard: 40, maxCard: 104, maxLines: 2 };
  const ROW_GAP = 6;

  // Each half divides its height between its two rows by what they hold.
  let theirRows = $state({ width: 0, height: 0 });
  let yourRows = $state({ width: 0, height: 0 });
  function split(side: typeof view.you, size: { width: number; height: number }): HalfSplit | null {
    if (size.width <= 0 || size.height <= 0) return null;
    return splitHalf(pileShapes(side.creatures), pileShapes(side.support), size.width - 22, size.height - 16,
      { gap: ROW_GAP, creatures: CREATURES, support: SUPPORT });
  }
  const theirSplit = $derived(split(board.opponent, theirRows));
  const yourSplit = $derived(split(board.you, yourRows));
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
    endReadOut();
    menuAt = null;
    stopNote = null;
    hush();
    since = { revision: frame.revision, frame, logLength: log.length };
    onact(offer);
  }

  function passTurn(): void {
    if (!canPassTurn) return;
    since = { revision: frame.revision, frame, logLength: log.length };
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
  function activate(title: string, offers: PlacedOffer[], element: HTMLElement): void {
    hush();
    if (!live || !offers.length) return;
    if (offers.length === 1) { act(offers[0]); return; }
    opener = element;
    const rect = element.getBoundingClientRect();
    menuAt = { title, offers, x: Math.min(rect.left, window.innerWidth - 300), y: rect.bottom + 6 > window.innerHeight - 160 ? Math.max(8, rect.top - 8 - 36 * offers.length - 30) : rect.bottom + 6 };
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
  const cards = $derived(new Map(inspectableCards([boardProjection.agent, boardProjection.opponent]).map(card => [card.id, card])));
  const shown = $derived(inspectMode === 'dock' ? docked : hovered?.subject ?? null);
  const shownPermanent = $derived(shown?.permanent != null ? permanents.get(shown.permanent) ?? null : null);
  const shownCard = $derived(shown?.card != null ? cards.get(shown.card) ?? null : null);

  const castTerms = $derived(permissionTerms(shownCard,
    id => permanents.get(id)?.name ?? `permanent #${id}`,
    id => id === boardProjection.agent.id ? you.name : opponent.name));

  function onPermanent(item: PermanentItem, element: HTMLElement): void {
    if (onDraft(item.permanent.id)) return;
    if (drafting || running) return;
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
  const trayReserve = $derived(view.tray.length && !finished ? Math.min(Math.floor(handWidth * 0.45), view.tray.length * (trayCard + 10) + 30) : 0);
  const fan = $derived(handFan(board.hand.length, handWidth - HAND_LABEL_WIDTH - 24 - trayReserve, { min: HAND_FLOOR, max: handFull }, 10));
  // Shrunk cards give the strip's spare height back to the board.
  const handTrack = $derived(board.hand.length === 0 && trayReserve === 0 ? '30px' : `${Math.min(handHeight, Math.ceil(fan.card / CARD_RATIO) + 22)}px`);

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
          { key: 'Space', does: 'The primary choice (Pass, Resolve, Done…)' },
          { key: '1–9', does: 'The numbered choice' },
          { key: 'Enter', does: 'Act on the focused card or choice' },
          { key: 'Tab', does: 'Next region: fields, hand, stack, choices, moves' },
          { key: '← → ↑ ↓', does: 'Move within a region' },
          { key: 'Shift Enter', does: 'Pass turn: pass every priority window until it ends (also F6)' },
          { key: 'Ctrl', does: 'Pause Auto-tap for the next payment (Full Control)' },
          { key: 'Shift Ctrl', does: 'Hold Full Control across turns' },
        ]),
    { key: 'Esc', does: recording ? 'Close a menu or preview' : "Close a menu or preview; skip the opponent's moves" },
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
    if (recording || !live || readingOut || result || drafting || !onautotap) return;
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

  /** Keys that move focus or open help during a read-out. */
  const MOVES_FOCUS = new Set(['Tab', 'Shift', '?', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']);
  let skippedKey: string | null = null;
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
    if (!event.repeat) skippedKey = null;
    const key = event.code || event.key;
    if (event.repeat && key === skippedKey) { event.preventDefault(); return; }
    // Enter on a focused choice, Space, Esc or a number key ends a read-out and acts on nothing.
    if (readingOut && !MOVES_FOCUS.has(event.key)) {
      event.preventDefault();
      skippedKey = key;
      endReadOut();
      return;
    }
    if (event.key === '?') { event.preventDefault(); keysOpen = !keysOpen; return; }
    if (event.key === 'Escape' && keysOpen) { closeKeys(); return; }
    if (event.key === 'Escape' && anyTrackerOpen) { myDeckOpen = false; oppDeckOpen = false; return; }
    if (event.key === 'Escape' && running) { stopSending(); return; }
    if (event.key === 'Escape' && armed !== null) { armed = null; return; }
    if (event.key === 'Escape') { closeMenu(); zoneAt = null; hovered = null; stopDwell(); return; }
    if (event.key === 'F6' || (event.key === 'Enter' && event.shiftKey)) { event.preventDefault(); passTurn(); return; }
    if (/^[1-9]$/.test(event.key)) {
      const placed = (menuAt?.offers ?? view.list).find(candidate => candidate.number === Number(event.key)) ?? null;
      if (placed && live) { event.preventDefault(); act(placed); }
      return;
    }
    // An open card menu takes Space as a press of its focused item.
    if (event.key === ' ' && menuAt) return;
    if (event.key === ' ' && !ownsSpace(event.target) && drafting) { event.preventDefault(); confirmDraft(); return; }
    if (event.key === ' ' && !ownsSpace(event.target) && view.primary && live) {
      event.preventDefault();
      act(view.primary);
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

<svelte:window onkeydown={onKeydown} onkeyup={onKeyup} onblur={() => { ctrlPending = false; ctrlShift = false; }} onpointermove={onPointerMove}
  onpointerdown={event => { if (keysOpen && !(event.target instanceof Element && event.target.closest('.keys, .keys-hint'))) keysOpen = false; }} />

<div class="game" class:with-sidebar={sidebarOpen} data-testid="game" data-action-space={frame.action_space} data-live={live} data-thinking={thinking !== null}
  data-turn={turnNumber} data-step={turnStep} data-turn-owner={yourTurn ? 'you' : 'opponent'} style:--plate={`${PLATE_WIDTH}px`} style:--hand-label={`${HAND_LABEL_WIDTH}px`} bind:this={root}>
  <header class="topbar">
    <a href="/games" class="brand" aria-label="Academy · Games">Academy</a>
    <p class="matchup type-caption" data-testid="deck-names">
      <span class="text-ink">{you.name}</span>{#if you.deck}<span class="text-ink-3">{' · '}{you.deck}</span>{/if}
      <span class="text-ink-3">{' vs '}</span>
      <span class="text-ink" title={[opponent.name, opponent.description].filter(Boolean).join(' — ')}>{said}</span>{#if opponent.deck}<span class="text-ink-3">{' · '}{opponent.deck}</span>{/if}
    </p>
    <p class="turn" data-testid="turn-number">Turn {turnNumber} · {yourTurn ? 'yours' : 'theirs'}{#if turnStep !== null && boardProjection.turn.creature_died_this_turn} · A creature died this turn{/if}</p>
    <div class="right">
      {#if connection}
        <!-- Connected says nothing: only a problem shows; the status stays for assistive technology. -->
        <span data-testid="connection-badge" data-connection-state={connection} role="status" aria-live="polite" aria-atomic="true"
          aria-label={`Connection status: ${connectionLabel}`} class="connection type-label" class:sr-only={connection === 'connected'}>{connection === 'connected' ? connectionLabel : connectionLabel.charAt(0).toUpperCase() + connectionLabel.slice(1)}</span>
      {/if}
      <button type="button" class="keys-hint type-caption" data-testid="keys-hint" aria-haspopup="dialog" aria-expanded={keysOpen}
        aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)" bind:this={keysButton} onclick={() => (keysOpen = !keysOpen)}><kbd>?</kbd></button>
      {#if !recording && ontoggleautotap}
        <button type="button" class="inspect-toggle type-caption" data-testid="auto-tap-toggle"
          aria-pressed={fullControl !== 'off' ? false : autoTap}
          data-full-control={fullControl}
          aria-label={fullControl === 'hold' ? 'Full Control (hold) — click to resume Auto-tap' : fullControl === 'once' ? 'Full Control (next payment) — click to resume Auto-tap' : autoTap ? 'Auto-tap is on — press Ctrl to pause once, Shift+Ctrl to hold' : 'Auto-tap is off — click to turn on'}
          title={fullControl === 'hold' ? 'Full Control (hold) · Shift+Ctrl or click to resume Auto-tap' : fullControl === 'once' ? 'Full Control (next payment) · Ctrl or click to resume Auto-tap' : autoTap ? 'Auto-tap ON · Ctrl pauses once · Shift+Ctrl holds Full Control' : 'Auto-tap OFF · click to auto-tap mandatory mana payments'}
          onclick={ontoggleautotap}>{fullControl === 'hold' ? 'Full control · hold' : fullControl === 'once' ? 'Full control' : autoTap ? 'Auto-tap' : 'Manual tap'}</button>
      {/if}
      <button type="button" class="inspect-toggle type-caption" data-testid="deck-tracker-toggle" aria-pressed={anyTrackerOpen}
        aria-label="Toggle decklist trackers" title="Show or hide Untapped-style deck trackers for both seats"
        onclick={toggleBothTrackers}>Decks</button>
      <button type="button" class="inspect-toggle type-caption" data-testid="inspect-toggle" aria-pressed={inspectMode === 'dock'}
        aria-label="Dock inspected cards in the sidebar" title={inspectMode === 'dock' ? 'Inspected cards show in the sidebar; press to show them beside the card' : 'Inspected cards show beside the card; press to dock them in the sidebar'}
        onclick={toggleInspect}>Dock</button>
      <span class="theme-slot"><ThemeSwitch /></span>
      {#if menu}{@render menu()}{/if}
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

  <div class="board" id="board" data-testid="game-board" aria-label="Board" bind:clientHeight={boardHeight} style:grid-template-rows={`minmax(0, 1fr) minmax(0, 1fr) ${handTrack}`}
    data-board={beatBoard ? 'step' : 'frame'}>
    <section use:region={{ across: connection !== null }} class="half opponent" aria-label={`${possessive(said)} battlefield`}>
      <Plate player={boardProjection.opponent} name={said} caption={[decidesBy(opponent.name), opponent.deck].filter(Boolean).join(' · ') || null} offers={board.opponent.offers} {live} damage={damage.players.get(boardProjection.opponent.player_index) ?? 0}
        choosing={opponentChoosing} side="opponent" onactivate={element => activate(said, board.opponent.offers, element)} onzone={(zone, cards, element) => openZone(`${said} · ${zone}`, cards, element)}
        ondeck={() => { oppDeckOpen = !oppDeckOpen; }} deckOpen={oppDeckOpen} />
      <div class="rows" bind:clientWidth={theirRows.width} bind:clientHeight={theirRows.height} style:grid-template-rows={rowsStyle(theirSplit, theirRows, false)}>
        <div class="row support"><Field piles={board.opponent.support} label={`${possessive(said)} lands and other permanents`} limits={SUPPORT} {live} asked={drafting ? new Set<number>() : board.asked} draft={draftMarks} touched={recent.touched} beat={beatSubjects} damage={damage.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} toward="down" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row creatures"><Field piles={board.opponent.creatures} label={`${possessive(said)} creatures`} limits={CREATURES} {live} asked={drafting ? new Set<number>() : board.asked} draft={draftMarks} touched={recent.touched} beat={beatSubjects} damage={damage.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} toward="down" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section use:region={{ across: connection !== null }} class="half you" aria-label="Your battlefield">
      <Plate player={boardProjection.agent} name={you.name} caption={you.deck} offers={board.you.offers} {live} damage={damage.players.get(boardProjection.agent.player_index) ?? 0}
        side="you" onactivate={element => activate(you.name, board.you.offers, element)} onzone={(zone, cards, element) => openZone(`Your ${zone.toLowerCase()}`, cards, element)}
        ondeck={selfTracker ? () => { myDeckOpen = !myDeckOpen; } : undefined} deckOpen={myDeckOpen} />
      <div class="rows" bind:clientWidth={yourRows.width} bind:clientHeight={yourRows.height} style:grid-template-rows={rowsStyle(yourSplit, yourRows, true)}>
        <div class="row creatures"><Field piles={board.you.creatures} label="Your creatures" limits={CREATURES} {live} asked={drafting ? new Set<number>() : board.asked} draft={draftMarks} touched={recent.touched} beat={beatSubjects} damage={damage.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} {acting} toward="up" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row support"><Field piles={board.you.support} label="Your lands and other permanents" limits={SUPPORT} {live} asked={drafting ? new Set<number>() : board.asked} draft={draftMarks} touched={recent.touched} beat={beatSubjects} damage={damage.objects} planned={payPlanned?.sources ?? null} {unpiled} {targeted} {acting} toward="up" names={boardNames} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section use:region={{ across: connection !== null }} class="hand" aria-label={`Your hand, ${board.hand.length} cards`} bind:clientWidth={handWidth}>
      <span class="type-heading hand-label text-ink-2">Hand · {board.hand.length}</span>
      <div class="hand-cards" style:width={`${fan.card + Math.max(0, board.hand.length - 1) * fan.step}px`}>
        {#each board.hand as item, index (item.card.id)}
          <div class="hand-card" class:playable={live && item.offers.length > 0} style:left={`${index * fan.step}px`}>
            <BoardCard name={item.card.name} width={fan.card} slice={fan.covered && index < board.hand.length - 1 ? fan.step : null} card={item.card} offers={item.offers} {live} asked={board.asked.has(item.card.id)}
              dim={acting && !openingHandDecision && item.offers.length === 0 ? 'hand' : null}
              onactivate={element => activate(item.card.name, item.offers, element)}
              onpreview={element => previewCard(element ? item.card : null, element)} />
          </div>
        {/each}
      </div>
      {#if view.tray.length && !finished}
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
  </div>

  <div class="rail-slot" id="rail-choices">
    <Rail {view} {frame} {live} {line} {hint} {primaryText} {status} {whose} {thinking} {opponentChoosing} {workStarted} {interrupted} {canPassTurn} {recent} {moves}
      opponentName={said} onact={act} onpassturn={passTurn} {paymentNote} onpointoffer={placed => (pointed = placed)}
      onstack={onStack} onpreviewname={previewName} {notices} over={finished}
      {evidence} {moveMarks} {shownMoves} {recapTitle} {onjump} recording={readOnly} {currentMove} reading={readingOut ? reading : undefined}
      draft={drafting || running ? combatControls : undefined} stopped={stopNote} />
  </div>

  {#snippet combatControls()}
    <div class="combat-draft" data-testid="combat-draft" data-kind={preview?.kind ?? running?.kind}>
      {#if running}
        <button type="button" class="btn btn-secondary" data-testid="combat-stop" aria-disabled={readingOut || undefined} onclick={stopSending}><kbd>Esc</kbd>Stop sending</button>
      {:else if words}
        <button type="button" class="btn btn-primary primary" data-testid="primary-action" data-draft-ready={exact !== null && !exact.refused}
          aria-label={`Confirm: ${exact?.refused ? exact.refused.reason : words.plan}`} title={words.plan}
          disabled={!exact || exact.refused !== null} aria-disabled={readingOut || undefined} onclick={confirmDraft}>
          <span>{words.text}</span><kbd>Space</kbd>
        </button>
        {#each counts as group (group.ids[0])}
          <div class="pile-count" data-testid="pile-count" role="group" aria-label={`${group.name}: ${group.drafted} of ${group.ids.length} attacking`}>
            <span class="type-caption"><b>{group.drafted}</b> of {group.ids.length} {group.name}</span>
            <button type="button" class="step" aria-disabled={readingOut || undefined} aria-label={`One fewer ${group.name}`} disabled={group.drafted === 0}
              onclick={() => { draft = setPileCount(draft, group.ids, group.drafted - 1); }}>−</button>
            <button type="button" class="step" aria-disabled={readingOut || undefined} aria-label={`One more ${group.name}`} disabled={group.drafted === group.ids.length}
              onclick={() => { draft = setPileCount(draft, group.ids, group.drafted + 1); }}>+</button>
          </div>
        {/each}
        {#if draft.attackers.length || draft.blocks.length}
          <button type="button" class="btn btn-secondary secondary" data-testid="combat-clear" aria-disabled={readingOut || undefined} onclick={() => { draft = EMPTY_DRAFT; armed = null; }}>Clear the draft</button>
        {/if}
        <p class="type-caption text-ink-3">Nothing is declared until you confirm{preview?.kind === 'blockers' ? '. Click a blocker, then the attacker it blocks' : ''}.</p>
      {/if}
    </div>
  {/snippet}

  {#snippet reading()}
    {#if beat}
      <button type="button" class="beat" data-testid="beat" aria-live="polite" onclick={endReadOut}
        title="Show your decision (Space)">
        <span class="beat-text">{beat.text}</span>
        <span class="beat-count type-caption">{beatAt + 1} of {beats.length} · Space to skip</span>
      </button>
    {/if}
  {/snippet}

  {#snippet stripToggle()}
    <!-- Live play's strip only counts, so Tab passes it by; a recording's opens to a navigable line. -->
    <button type="button" class="strip-toggle" aria-expanded={strip!.open} onclick={strip!.ontoggle} data-testid="strip-toggle"
      tabindex={strip!.onjump ? undefined : -1}>
      <span aria-hidden="true">{strip!.open ? '▾' : '▸'}</span> Decisions · {strip!.marks.filter(mark => mark.decision).length}
    </button>
  {/snippet}
  {#if bottom || strip?.open}
    <div class="bottom">
      {@render bottom?.()}
      {#if strip?.open}<Strip marks={strip.marks} selected={strip.selected} onjump={strip.onjump} note={strip.note} scale={strip.scale} />{/if}
    </div>
  {/if}
  <nav class="stops-slot" aria-label={`Turn ${turnNumber}, ${yourTurn ? 'yours' : "opponent's"}: steps and stops`}>
    <StopBar {track} oncycle={oncyclestop} aside={strip ? stripToggle : undefined} />
  </nav>
  <Lines {lines} {root} revision={frame.revision} />
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
      <Inspect name={docked.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? docked.text} permanent={shownPermanent} card={shownCard} {castTerms}
        blocks={shownPermanent?.blocking !== undefined ? names.get(shownPermanent.blocking) ?? null : null} art={72} />
    {:else}
      <p class="type-body text-ink-3">Hover or focus a card to inspect it.</p>
    {/if}
  </section>
{/snippet}

{#if inspectMode === 'float' && hovered}
  <div class="preview" aria-hidden="true" style:left={`${hovered.x}px`} style:top={`${hovered.y}px`} data-testid="card-preview">
    <Inspect name={hovered.subject.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? hovered.subject.text} permanent={shownPermanent} card={shownCard} {castTerms}
      blocks={shownPermanent?.blocking !== undefined ? names.get(shownPermanent.blocking) ?? null : null} />
  </div>
{/if}

<style>
  /*
   * The board's one layering order, lowest first: cards, the lines between
   * cards and their targets, floating panels (the result card).
   * Each row of permanents and the hand keep their own stacking, so a raised,
   * attacking or lifted card orders only among its neighbours and never climbs
   * over a line or a panel. Above the whole game (itself at 10) float the inspect
   * preview (250), then card menus and the keys sheet (300). Add no other z-index.
   */
  .game {
    --layer-lines: 1; --layer-panel: 2;
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
  /* The inspect pane takes the height of its card, to two fifths of the column; the trackers share the rest and scroll inside. */
  .sidebar > :global(*) { flex: 1 1 0; min-height: 0; }
  .sidebar > :global(* + *) { border-top: 1px solid var(--border-strong); }
  .sidebar > .dock { flex: none; max-height: 40%; padding: 10px 14px; }
  .topbar {
    grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 0 14px;
    border-bottom: 1px solid var(--border); background: var(--bg-surface); min-width: 0;
  }
  .brand { font-family: var(--font-serif); font-weight: 600; font-size: 20px; line-height: 28px; color: var(--accent-text); }
  .matchup { min-width: 0; flex: 0 1 auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* The matchup gives way first: the turn and the controls keep their width. */
  .turn { margin: 0 auto; flex-shrink: 0; font-size: 14px; line-height: 20px; font-weight: 600; white-space: nowrap; }
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
  @keyframes appear { from { opacity: 0; } }
  /* A beat heads the rail's decision block, above the prompt that waits for it. */
  .beat {
    display: grid; gap: 2px; justify-items: start; width: 100%; padding: 6px 10px; border: 1px solid var(--border-strong);
    border-left: 3px solid var(--accent); border-radius: var(--radius-sm); background: var(--bg-field); text-align: left; cursor: pointer;
    animation: appear var(--motion-fast) ease-out;
  }
  .beat-text { font-size: 14px; line-height: 20px; color: var(--text); display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 3; line-clamp: 3; overflow: hidden; }
  .beat-count { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
  .result-card.folded { top: auto; bottom: 12px; left: auto; right: 12px; transform: none; width: auto; padding: 6px; }
  .result-card .fold { justify-self: end; color: var(--text-secondary); }
  .result-card .fold:hover { color: var(--text); }

  .combat-draft { display: grid; gap: 6px; }
  /* The draft's buttons are the rail's primary and secondary, drawn by the board. */
  .combat-draft .primary { height: 40px; padding-block: 6px; width: 100%; font-size: 14px; justify-content: center; white-space: nowrap; }
  .combat-draft .primary span { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .combat-draft .primary kbd { flex: none; opacity: 1; }
  .combat-draft .primary:disabled { background: var(--bg-muted); color: var(--text-secondary); opacity: 1; }
  /* A read-out holds the draft's controls as Rail holds its own: listed, dimmed, a press only ends it. */
  .combat-draft [aria-disabled] { opacity: 0.55; cursor: default; }
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
  .theme-slot { display: inline-flex; color: var(--text-tertiary); }
  .theme-slot:hover { color: var(--text); }
  .inspect-toggle { color: var(--text-tertiary); padding: 2px 6px; border-radius: var(--radius-sm); white-space: nowrap; }
  .inspect-toggle:hover { color: var(--text); background: var(--bg-field); }
  .inspect-toggle[aria-pressed='true'] { color: var(--text); background: var(--bg-field); }

  @media (prefers-reduced-motion: reduce) {
    .hand-card, .board { transition: none; }
    .menu, .preview, .result-card, .keys { animation: none; }
  }
</style>
