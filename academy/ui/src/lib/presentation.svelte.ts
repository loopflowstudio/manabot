import {
  copyPresentationEvents,
  mergePresentationLabels,
  presentationEventLabels,
  type PresentationEvent,
  type PresentationLabels,
  type PresentationSequence,
} from './presentation';

function emptyLabels(): PresentationLabels {
  return { objects: {}, players: {}, stacks: {} };
}

/** Viewer-scoped validated presentation track store. */
export class PresentationPlayer {
  scope = $state('');
  events = $state<PresentationEvent[]>([]);
  private eventLabels = $state<PresentationLabels[]>([]);
  private retainedLabels = $derived(mergePresentationLabels(...this.eventLabels));
  private missingPrefix = $state(false);
  incomplete = $state(false);

  get labels(): PresentationLabels {
    return this.retainedLabels;
  }

  labelsAt(index: number): PresentationLabels {
    return this.eventLabels[index] ?? emptyLabels();
  }

  get unavailablePrefix(): boolean {
    return this.missingPrefix || (this.events[0]?.seq ?? 0) > 0;
  }

  /** Match/viewer/branch identity must be set before accepting any names. */
  setScope(scope: string): void {
    if (scope === this.scope) return;
    this.clear();
    this.scope = scope;
  }

  load(sequence: PresentationSequence): void {
    this.loadEvents(sequence.events, sequence.labels, sequence.id);
  }

  /** Replace a recorded sequence. */
  loadEvents(
    events: readonly PresentationEvent[],
    labels: PresentationLabels,
    scope = 'recorded',
  ): void {
    this.loadPrefix([{ events, labels }], scope);
  }

  /** Canonical seek rebuilds exactly the selected prefix. */
  loadPrefix(
    batches: readonly { events: readonly PresentationEvent[]; labels: PresentationLabels }[],
    scope: string,
  ): void {
    this.clear();
    this.scope = scope;
    for (const batch of batches) {
      this.append(batch.events, batch.labels);
    }
  }

  private append(events: readonly PresentationEvent[], labels: PresentationLabels): void {
    const incoming = copyPresentationEvents($state.snapshot(events));
    const last = this.events.at(-1);
    if (last && incoming[0] && incoming[0].seq <= last.seq) {
      throw new Error(
        `Live presentation batch starts at ${incoming[0].seq}, not after existing event ${last.seq}`,
      );
    }
    // Pre-transition names can name departed subjects, but never overwrite
    // snapshots already retained for earlier facts.
    const available = mergePresentationLabels(this.labels, $state.snapshot(labels));
    this.events.push(...incoming);
    this.eventLabels.push(...incoming.map(event => presentationEventLabels(event, available)));
  }

  /** Frame authority has already committed before this live admission. */
  enqueue(events: readonly PresentationEvent[], labels: PresentationLabels): void {
    this.append(events, labels);
  }

  /** Same-scope recovery preserves retained history and merges unseen facts by seq. */
  recover(events: readonly PresentationEvent[], labels: PresentationLabels, cursor = 0): void {
    const validated = copyPresentationEvents($state.snapshot(events));
    const seen = new Set(this.events.map(event => event.seq));
    const unseen = validated.filter(event => !seen.has(event.seq));
    const available = mergePresentationLabels(this.labels, $state.snapshot(labels));
    const rows = this.events.map((event, index) => ({ event, labels: this.labelsAt(index) }));
    rows.push(...unseen.map(event => ({ event, labels: presentationEventLabels(event, available) })));
    rows.sort((left, right) => left.event.seq - right.event.seq);
    this.events = rows.map(row => row.event);
    this.eventLabels = rows.map(row => row.labels);
    this.missingPrefix = this.events.length === 0 && cursor > 0;
  }

  reject(): void {
    this.incomplete = true;
  }

  clear(): void {
    this.events = [];
    this.eventLabels = [];
    this.incomplete = false;
    this.missingPrefix = false;
  }
}

export function createPresentationPlayer(): PresentationPlayer {
  return new PresentationPlayer();
}

/** Live board player fed only by validated protocol updates and recovery tails. */
export const presentationPlayer = createPresentationPlayer();

/** Isolated shared-table Study track; live updates continue on the board player. */
export const studyPresentationPlayer = createPresentationPlayer();
