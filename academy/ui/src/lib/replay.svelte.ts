import { recordFetch } from './records';
import type { WorkStatus } from './types';
import type { RecordedPosition, RecordingIndex } from './recording';

/** One read cursor for all recordings; producing is an explicit edge capability.
 * Selection generations reject reads after a seek or reader pause. A separate
 * source generation lets a pending producer refresh only its own recording.
 */
export class ReplayStore {
  index = $state<RecordingIndex | null>(null);
  current = $state<RecordedPosition | null>(null);
  selected = $state(0);
  playing = $state(false);
  speed = $state(1);
  busy = $state(false);
  producing = $state(false);
  error = $state('');
  headUncertain = $state(false);
  work = $state<WorkStatus | null>(null);
  workDisconnected = $state(false);
  reveal = $state(false);
  nearSeat = $state(0);
  private generation = 0;
  private id = '';
  private sourceGeneration = 0;
  private productionGeneration = 0;
  private stopWorkPoll = (): void => {};

  request(path: string, init: RequestInit = {}): Promise<Response> {
    const token = sessionStorage.getItem(`etude.watch.${this.id}`);
    const headers = new Headers(init.headers);
    if (token) headers.set('x-etude-participant-token', token);
    return recordFetch(path, {...init, headers});
  }

  async api<T>(path: string, body?: unknown): Promise<T> {
    const response = await this.request(path, {
      headers: { 'content-type': 'application/json' },
      ...(body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      if (typeof payload.detail?.message === 'string') throw new Error(payload.detail.message);
      throw new Error(typeof payload.detail === 'string' ? payload.detail.replaceAll('_', ' ') : `Request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }

  private async readIndex(id: string): Promise<RecordingIndex> {
    const index = await this.api<RecordingIndex>(`/api/traces/${encodeURIComponent(id)}/recording`);
    while (index.next_offset !== null) {
      const page = await this.api<RecordingIndex>(`/api/traces/${encodeURIComponent(id)}/recording?offset=${index.next_offset}`);
      if (page.revision !== index.revision || page.total !== index.total) throw new Error('The recording changed while reading. Refresh to read the saved head.');
      index.positions.push(...page.positions);
      index.next_offset = page.next_offset;
    }
    return index;
  }

  async open(id: string, key?: string): Promise<void> {
    const generation = ++this.generation;
    this.stopWorkPoll();
    ++this.sourceGeneration;
    this.workDisconnected = false; this.work = null; this.id = id; this.playing = false; this.busy = true; this.current = null; this.index = null; this.producing = false; this.headUncertain = false; this.error = '';
    try {
      const index = await this.readIndex(id);
      if (generation !== this.generation) return;
      this.index = index;
      const selected = key ? index.positions.findIndex(position => position.key === key) : Math.max(0, index.positions.findIndex(position => position.kind === 'choice'));
      if (selected < 0) { this.error = 'Saved position is unavailable. Select an available position.'; this.busy = false; return; }
      await this.seek(selected);
    } catch (reason) { if (generation === this.generation) this.fail(reason); }
    finally { if (generation === this.generation) this.busy = false; }
  }

  async seek(selected: number, animate = false): Promise<void> {
    if (!this.index) return;
    if (!animate) this.playing = false;
    const position = this.index.positions[selected];
    if (!position) return;
    const generation = ++this.generation;
    this.busy = true; this.error = '';
    try {
      // Reveal is a per-choice capability. Consequences never borrow it.
      const reveal = this.reveal && position.can_reveal;
      let current = await this.api<RecordedPosition>(`/api/traces/${encodeURIComponent(this.id)}/positions/${encodeURIComponent(position.key)}?reveal=${reveal}`);
      if (generation !== this.generation) return;
      // A large transition can span history pages. Retain every ordered beat
      // of this action before playback, with bounded requests and stale guards.
      while (current.kind === 'consequences' && current.history_before !== null && (current.history[0]?.seq ?? current.cursor) > current.transition_start) {
        const earlier = await this.api<RecordedPosition>(`/api/traces/${encodeURIComponent(this.id)}/positions/${encodeURIComponent(position.key)}?before=${current.history_before}`);
        if (generation !== this.generation) return;
        current = {...current, history: [...earlier.history, ...current.history], history_labels: [...earlier.history_labels, ...current.history_labels], history_before: earlier.history_before};
      }
      this.current = current; this.selected = selected;
    } catch (reason) { if (generation === this.generation) this.fail(reason); }
    finally { if (generation === this.generation) this.busy = false; }
  }

  async produce(retry = false): Promise<void> {
    if (!this.index?.can_produce || this.producing || this.busy || this.selected !== this.index.positions.length - 1) return;
    if (this.headUncertain || (this.index.mirror_retry && !retry)) { this.pause(); return; }
    this.producing = true; this.error = '';
    const oldLength = this.index.positions.length;
    const revision = this.index.revision;
    const generation = this.generation;
    const sourceGeneration = this.sourceGeneration;
    const id = this.id;
    const production = ++this.productionGeneration;
    this.work = null; this.workDisconnected = false;
    const isCurrentProduction = (): boolean => sourceGeneration === this.sourceGeneration
      && production === this.productionGeneration && this.producing;
    let polling = false;
    const poll = async (): Promise<void> => {
      if (polling) return;
      polling = true;
      try {
        const {work} = await this.api<{work: WorkStatus | null}>(`/api/watch/${encodeURIComponent(id)}/work`);
        if (!isCurrentProduction()) return;
        this.workDisconnected = false;
        if (work?.scope === 'match' && work.match_id === id
          && work.revision === revision && work.sequence >= (this.work?.sequence ?? 0)) this.work = work;
      } catch {
        // A failed read proves no outcome and grants no retry. Stop claiming
        // verified remote progress until a subsequent status read succeeds.
        if (isCurrentProduction()) this.workDisconnected = true;
      }
      finally { polling = false; }
    };
    const timer = setInterval(() => void poll(), 1000);
    const stopPoll = (): void => clearInterval(timer);
    this.stopWorkPoll = stopPoll;
    try {
      await this.api(`/api/watch/${encodeURIComponent(this.id)}/step`, {revision: this.index.revision, ...(retry ? {retry: true} : {})});
      if (sourceGeneration !== this.sourceGeneration) return;
      const index = await this.readIndex(id);
      if (sourceGeneration !== this.sourceGeneration) return;
      this.index = index;
      this.producing = false;
      if (generation !== this.generation) return;
      // Latest was replaced by the newly appended choice, never revision - 1.
      await this.seek(Math.min(oldLength - 1, this.index.positions.length - 1), true);
    } catch (reason) {
      if (sourceGeneration !== this.sourceGeneration) return;
      this.fail(reason);
      this.headUncertain = true;
      // A lost reply may already have committed. Read the head, never retry here.
      try {
        const index = await this.readIndex(id);
        if (sourceGeneration === this.sourceGeneration) {
          this.index = index;
          this.headUncertain = false;
        }
      }
      catch { /* Preserve the original failure and the retained board. */ }
      if (sourceGeneration !== this.sourceGeneration) return;
      this.error += ' Refresh the saved head before explicitly continuing.';
    } finally {
      stopPoll();
      if (sourceGeneration === this.sourceGeneration) { this.producing = false; this.work = null; }
    }
  }

  async refreshHead(): Promise<void> {
    if (this.producing || this.busy) return;
    const generation = this.sourceGeneration;
    this.busy = true;
    try {
      const index = await this.readIndex(this.id);
      if (generation !== this.sourceGeneration) return;
      this.index = index; this.headUncertain = false; this.error = '';
    } catch (reason) { if (generation === this.sourceGeneration) this.fail(reason); }
    finally { if (generation === this.sourceGeneration) this.busy = false; }
  }

  async next(animate = false): Promise<void> {
    if (!this.index || this.busy) return;
    if (this.producing && this.selected + 1 >= this.index.positions.length) return;
    // A paused producer can replace the old latest marker with a saved choice.
    if (this.current?.key !== this.index.positions[this.selected]?.key) await this.seek(this.selected, animate);
    else if (this.selected + 1 < this.index.positions.length) await this.seek(this.selected + 1, animate);
    else if (this.index.can_produce) await this.produce();
    else this.playing = false;
  }
  pause(): void {
    ++this.generation;
    this.playing = false;
    this.busy = false;
  }
  cancelReads(): void { this.stopWorkPoll(); ++this.sourceGeneration; this.workDisconnected = false; this.work = null; this.producing = false; this.pause(); }
  private fail(reason: unknown): void { this.error = reason instanceof Error ? reason.message : String(reason); this.playing = false; }
}
export function createReplayStore(): ReplayStore { return new ReplayStore(); }
