import {
  copyPresentationEvents,
  presentationBeat,
  type PresentationBeat,
  type PresentationEvent,
  type PresentationLabels,
  type PresentationSequence,
} from './presentation';

const NORMAL_SPEED = 1;
const FAST_SPEED = 4;
const REDUCED_MOTION_BEAT_MS = 100;

function emptyLabels(): PresentationLabels {
  return { objects: {}, players: {}, stacks: {} };
}

/**
 * Deterministic presentation cursor shared by live updates and replay loads.
 * The controller changes only optional theater; authoritative state is already
 * committed before events are enqueued.
 */
export class PresentationPlayer {
  events = $state<PresentationEvent[]>([]);
  labels = $state<PresentationLabels>(emptyLabels());
  currentIndex = $state(0);
  playing = $state(false);
  speed = $state(NORMAL_SPEED);
  reducedMotion = $state(false);

  get currentEvent(): PresentationEvent | null {
    return this.events[this.currentIndex] ?? null;
  }

  get currentBeat(): PresentationBeat | null {
    const event = this.currentEvent;
    return event ? presentationBeat(event, this.labels) : null;
  }

  get remaining(): number {
    return Math.max(0, this.events.length - this.currentIndex);
  }

  get effectiveDurationMs(): number {
    const event = this.currentEvent;
    if (!event) {
      return 0;
    }
    if (this.reducedMotion) {
      return REDUCED_MOTION_BEAT_MS;
    }
    return Math.max(80, Math.round(event.suggested_ms / this.speed));
  }

  /** Replay/recovery path: replace theater and seek to the first beat. */
  load(sequence: PresentationSequence): void {
    this.loadEvents(sequence.events, sequence.labels);
  }

  /** Replay path when events and viewer-safe labels come from persisted data. */
  loadEvents(
    events: readonly PresentationEvent[],
    labels: PresentationLabels,
  ): void {
    this.events = copyPresentationEvents(events);
    this.labels = structuredClone(labels);
    this.currentIndex = 0;
    this.speed = NORMAL_SPEED;
    this.playing = this.events.length > 0;
  }

  /** Live FrameUpdate path: append a strictly newer ordered event batch. */
  enqueue(events: readonly PresentationEvent[], labels: PresentationLabels): void {
    const incoming = copyPresentationEvents(events);
    const last = this.events.at(-1);
    const first = incoming[0];
    if (last && first && first.seq <= last.seq) {
      throw new Error(
        `Live presentation batch starts at ${first.seq}, not after existing event ${last.seq}`,
      );
    }

    this.events = [...this.events, ...incoming];
    this.labels = structuredClone(labels);
    if (incoming.length > 0) {
      this.playing = true;
    }
  }

  /** Recovery cancels speculative theater before an optional viewer-safe tail. */
  recover(
    events: readonly PresentationEvent[] = [],
    labels: PresentationLabels = emptyLabels(),
  ): void {
    this.clear();
    this.loadEvents(events, labels);
  }

  advance(): void {
    if (!this.currentEvent) {
      this.playing = false;
      return;
    }
    this.currentIndex += 1;
    if (!this.currentEvent) {
      this.playing = false;
      this.speed = NORMAL_SPEED;
    }
  }

  skipCurrent(): void {
    this.advance();
  }

  finishSequence(): void {
    this.currentIndex = this.events.length;
    this.playing = false;
    this.speed = NORMAL_SPEED;
  }

  setFastForward(enabled: boolean): void {
    this.speed = enabled ? FAST_SPEED : NORMAL_SPEED;
  }

  setReducedMotion(enabled: boolean): void {
    this.reducedMotion = enabled;
  }

  clear(): void {
    this.events = [];
    this.labels = emptyLabels();
    this.currentIndex = 0;
    this.playing = false;
    this.speed = NORMAL_SPEED;
  }
}

export function createPresentationPlayer(): PresentationPlayer {
  return new PresentationPlayer();
}

/** Live table player fed only by protocol FrameUpdates and recovery tails. */
export const presentationPlayer = createPresentationPlayer();
