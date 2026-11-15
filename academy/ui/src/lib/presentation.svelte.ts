import {
  copyPresentationEvents,
  mergePresentationLabels,
  presentationBeat,
  presentationEventLabels,
  type PresentationBeat,
  type PresentationEvent,
  type PresentationLabels,
  type PresentationSequence,
} from './presentation';

const NORMAL_SPEED = 1;
const FAST_SPEED = 4;
export const LIVE_BEAT_LIMIT = 12;
export const LIVE_DURATION_LIMIT_MS = 2000;

function emptyLabels(): PresentationLabels {
  return { objects: {}, players: {}, stacks: {} };
}

/** Viewer-scoped semantic history and an independent optional narration cursor.
 * Pending indices reference retained facts. Live admission budgets the whole
 * current beat conservatively, so even paused theater cannot accumulate debt.
 */
export class PresentationPlayer {
  scope = $state('');
  events = $state<PresentationEvent[]>([]);
  private eventLabels = $state<PresentationLabels[]>([]);
  private retainedLabels = $derived(mergePresentationLabels(...this.eventLabels));
  private pending = $state<number[]>([]);
  private live = true;
  private missingPrefix = $state(false);
  incomplete = $state(false);
  playing = $state(false);
  speed = $state(NORMAL_SPEED);
  reducedMotion = $state(false);

  get labels(): PresentationLabels {
    return this.retainedLabels;
  }

  labelsAt(index: number): PresentationLabels {
    return this.eventLabels[index] ?? emptyLabels();
  }

  get currentIndex(): number { return this.pending[0] ?? this.events.length; }
  get currentEvent(): PresentationEvent | null { return this.events[this.currentIndex] ?? null; }
  get currentBeat(): PresentationBeat | null {
    const event = this.currentEvent;
    return event ? presentationBeat(event, this.labelsAt(this.currentIndex)) : null;
  }
  get remaining(): number { return this.pending.length; }
  private duration(event: PresentationEvent): number {
    const duration = Math.max(80, Math.round(event.suggested_ms / this.speed));
    return this.live ? Math.min(LIVE_DURATION_LIMIT_MS, duration) : duration;
  }
  get pendingDurationMs(): number {
    return this.pending.reduce((sum, index) => sum + this.duration(this.events[index]), 0);
  }
  get effectiveDurationMs(): number {
    return this.currentEvent ? this.duration(this.currentEvent) : 0;
  }
  get unavailablePrefix(): boolean { return this.missingPrefix || (this.events[0]?.seq ?? 0) > 0; }

  /** Match/viewer/branch identity must be set before accepting any names. */
  setScope(scope: string): void {
    if (scope === this.scope) return;
    this.clear();
    this.scope = scope;
  }

  load(sequence: PresentationSequence): void {
    this.loadEvents(sequence.events, sequence.labels, sequence.id);
  }

  /** Replace a recorded sequence, with explicit uncapped ordered playback. */
  loadEvents(events: readonly PresentationEvent[], labels: PresentationLabels, scope = 'recorded'): void {
    this.loadPrefix([{ events, labels }], scope);
  }

  /** Canonical seek rebuilds exactly the selected prefix; only its last batch animates. */
  loadPrefix(batches: readonly { events: readonly PresentationEvent[]; labels: PresentationLabels }[], scope: string, animate = true): void {
    this.clear();
    this.scope = scope;
    this.live = false;
    for (const batch of batches) {
      const start = this.events.length;
      this.append(batch.events, batch.labels);
      this.pending = animate ? this.events.slice(start).map((_, offset) => start + offset) : [];
    }
    this.playing = this.remaining > 0;
  }

  /** The replay transport chooses which retained transition to narrate. */
  playRecordedFrom(seq: number): void {
    this.pending = this.events.flatMap((event, index) => event.seq >= seq ? [index] : []);
    this.playing = false;
  }

  private append(events: readonly PresentationEvent[], labels: PresentationLabels): void {
    const incoming = copyPresentationEvents($state.snapshot(events));
    const last = this.events.at(-1);
    if (last && incoming[0] && incoming[0].seq <= last.seq) {
      throw new Error(`Live presentation batch starts at ${incoming[0].seq}, not after existing event ${last.seq}`);
    }
    // Pre-transition names can name departed subjects, but never overwrite
    // snapshots already retained for earlier facts.
    const available = mergePresentationLabels(this.labels, $state.snapshot(labels));
    this.events.push(...incoming);
    this.eventLabels.push(...incoming.map(event => presentationEventLabels(event, available)));
  }

  private boundLiveQueue(): void {
    if (!this.live) return;
    let start = this.pending.length;
    let duration = 0;
    // Only inspect the newest bounded suffix, even for a large event batch.
    while (start > 0 && this.pending.length - start < LIVE_BEAT_LIMIT) {
      const nextDuration = this.duration(this.events[this.pending[start - 1]]);
      if (duration + nextDuration > LIVE_DURATION_LIMIT_MS) break;
      duration += nextDuration;
      start -= 1;
    }
    if (start > 0) this.pending = this.pending.slice(start);
    if (!this.remaining) this.playing = false;
  }

  /** Frame authority has already committed before this live admission. */
  enqueue(events: readonly PresentationEvent[], labels: PresentationLabels): void {
    const start = this.events.length;
    const wasIdle = this.remaining === 0;
    this.live = true;
    this.append(events, labels);
    this.pending.push(...this.events.slice(start).map((_, offset) => start + offset));
    if (events.length && wasIdle) this.playing = true;
    this.boundLiveQueue();
  }

  /** Same-scope recovery preserves reading, cancels stale motion, and animates only unseen facts. */
  recover(events: readonly PresentationEvent[], labels: PresentationLabels, cursor = 0): void {
    const validated = copyPresentationEvents($state.snapshot(events));
    const seen = new Set(this.events.map(event => event.seq));
    const unseen = validated.filter(event => !seen.has(event.seq));
    const available = mergePresentationLabels(this.labels, $state.snapshot(labels));
    const rows = this.events.map((event, index) => ({ event, labels: this.labelsAt(index) }));
    rows.push(...unseen.map(event => ({ event, labels: presentationEventLabels(event, available) })));
    rows.sort((left, right) => left.event.seq - right.event.seq);
    this.finishSequence();
    this.live = true;
    this.events = rows.map(row => row.event);
    this.eventLabels = rows.map(row => row.labels);
    this.missingPrefix = this.events.length === 0 && cursor > 0;
    this.pending = rows.flatMap((row, index) => seen.has(row.event.seq) ? [] : [index]);
    this.playing = this.remaining > 0;
    this.boundLiveQueue();
  }

  reject(): void {
    this.finishSequence();
    this.incomplete = true;
  }
  advance(): void {
    this.pending.shift();
    if (!this.remaining) {
      this.playing = false;
      this.speed = NORMAL_SPEED;
    }
  }
  togglePlayback(): void { if (this.currentEvent) this.playing = !this.playing; }
  skipCurrent(): void { this.advance(); }
  finishSequence(): void {
    this.pending = [];
    this.playing = false;
    this.speed = NORMAL_SPEED;
  }
  setFastForward(enabled: boolean): void {
    this.speed = enabled ? FAST_SPEED : NORMAL_SPEED;
    this.boundLiveQueue();
  }
  setReducedMotion(enabled: boolean): void { this.reducedMotion = enabled; }
  clear(): void {
    this.events = [];
    this.eventLabels = [];
    this.incomplete = false;
    this.missingPrefix = false;
    this.live = true;
    this.finishSequence();
  }
}

export function createPresentationPlayer(): PresentationPlayer { return new PresentationPlayer(); }
/** Live table player fed only by validated protocol updates and recovery tails. */
export const presentationPlayer = createPresentationPlayer();

/** Isolated shared-table Study cursor; live updates continue on the table player. */
export const studyPresentationPlayer = createPresentationPlayer();
