import { buildReplayFrames } from './replay';
import type { ReplayFrame, Trace, TraceSummary } from './types';

export class ReplayStore {
  summaries = $state<TraceSummary[]>([]);
  trace = $state<Trace | null>(null);
  frames = $state<ReplayFrame[]>([]);
  currentFrameIndex = $state(0);
  playing = $state(false);
  speed = $state(1);
  loadingList = $state(false);
  loadingTrace = $state(false);
  errorMessage = $state<string | null>(null);

  setSummaries(summaries: TraceSummary[]): void {
    this.summaries = summaries;
  }

  setLoadingList(next: boolean): void {
    this.loadingList = next;
  }

  setLoadingTrace(next: boolean): void {
    this.loadingTrace = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  loadTrace(trace: Trace): void {
    this.trace = trace;
    this.frames = buildReplayFrames(trace);
    this.currentFrameIndex = 0;
    this.playing = false;
    this.errorMessage = null;
  }

  currentFrame(): ReplayFrame | null {
    return this.frames[this.currentFrameIndex] ?? null;
  }

  setFrame(index: number): void {
    if (this.frames.length === 0) {
      this.currentFrameIndex = 0;
      this.playing = false;
      return;
    }

    const nextIndex = Math.max(0, Math.min(index, this.frames.length - 1));
    this.currentFrameIndex = nextIndex;
    if (nextIndex >= this.frames.length - 1) {
      this.playing = false;
    }
  }

  nextFrame(): void {
    this.setFrame(this.currentFrameIndex + 1);
  }

  previousFrame(): void {
    this.setFrame(this.currentFrameIndex - 1);
  }

  togglePlaying(): void {
    if (this.frames.length <= 1) {
      this.playing = false;
      return;
    }

    if (this.currentFrameIndex >= this.frames.length - 1) {
      this.playing = false;
      return;
    }

    this.playing = !this.playing;
  }

  setSpeed(speed: number): void {
    this.speed = speed;
  }

  tick(): void {
    if (!this.playing) {
      return;
    }

    this.nextFrame();
  }
}

export function createReplayStore(): ReplayStore {
  return new ReplayStore();
}
