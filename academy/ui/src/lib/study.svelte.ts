import {
  assertRestoredReplayDecisionBinds,
  assertViewerSafeReplayProjectionResponse,
  type AddressedReplayDecision,
  type CanonicalReplayProjectionResponseV1,
  type RestoredReplayDecision,
} from './replay-index';
import {
  bindStudyArtifact,
  buildStudyPlans,
  type StudyDisplayPlan,
  type StudyPlanKind,
  type StudyPlanPreviewResponse,
  type StudyRetryResponse,
  type StudyRevealResponse,
} from './study-runtime';
import type { Observation } from './types';

export type StudyPhase = 'score' | 'restored' | 'retried' | 'revealed';

function cloneDto<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export class StudyStore {
  projection = $state<CanonicalReplayProjectionResponseV1 | null>(null);
  selectedDecision = $state<AddressedReplayDecision | null>(null);
  restored = $state<RestoredReplayDecision | null>(null);
  phase = $state<StudyPhase>('score');
  attemptId = $state<string | null>(null);
  branchObservation = $state<Observation | null>(null);
  plans = $state<StudyDisplayPlan[]>([]);
  selectedPlan = $state<StudyPlanKind | null>(null);
  preview = $state<StudyPlanPreviewResponse | null>(null);
  loadingScore = $state(false);
  busy = $state(false);
  errorMessage = $state<string | null>(null);

  loadProjection(projection: CanonicalReplayProjectionResponseV1): void {
    assertViewerSafeReplayProjectionResponse(projection);
    this.projection = cloneDto(projection);
    this.resetSelection();
  }

  clearProjection(): void {
    this.projection = null;
    this.resetSelection();
  }

  setLoadingScore(next: boolean): void {
    this.loadingScore = next;
  }

  setBusy(next: boolean): void {
    this.busy = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  restore(
    decision: AddressedReplayDecision,
    restored: RestoredReplayDecision,
  ): void {
    if (!this.projection) throw new Error('Study Score is not loaded.');
    assertRestoredReplayDecisionBinds(this.projection, restored);
    if (decision.address !== restored.address) {
      throw new Error('Selected Study decision address drifted.');
    }
    this.selectedDecision = cloneDto(decision);
    this.restored = cloneDto(restored);
    this.phase = 'restored';
    this.attemptId = null;
    this.branchObservation = null;
    this.plans = [];
    this.selectedPlan = null;
    this.preview = null;
    this.errorMessage = null;
  }

  acceptRetry(response: StudyRetryResponse): void {
    if (!this.restored || response.address !== this.restored.address) {
      throw new Error('Retry response differs from the restored Study decision.');
    }
    this.attemptId = response.attempt_id;
    this.branchObservation = cloneDto(response.retry.projection);
    this.phase = 'retried';
    this.plans = [];
    this.selectedPlan = null;
    this.preview = null;
    this.errorMessage = null;
  }

  async reveal(response: StudyRevealResponse): Promise<void> {
    if (
      !this.projection
      || !this.restored
      || !this.attemptId
      || response.attempt_id !== this.attemptId
    ) {
      throw new Error('Reveal response differs from the active Study attempt.');
    }
    const landmark = await bindStudyArtifact(
      response.artifact,
      this.projection,
      this.restored,
    );
    this.plans = buildStudyPlans(landmark);
    this.phase = 'revealed';
    this.selectedPlan = 'played';
    this.preview = null;
    this.errorMessage = null;
  }

  acceptPreview(response: StudyPlanPreviewResponse): void {
    if (!this.attemptId || response.attempt_id !== this.attemptId) {
      throw new Error('Plan preview differs from the active Study attempt.');
    }
    const plan = this.plans.find(({ kind }) => kind === response.plan);
    if (
      !plan
      || plan.command.offer_id !== response.command.offer_id
      || plan.offer.id !== response.offer.id
    ) {
      throw new Error('Plan preview differs from the revealed Study plan.');
    }
    this.selectedPlan = response.plan;
    this.preview = cloneDto(response);
    this.errorMessage = null;
  }

  returnToScore(restored: RestoredReplayDecision): void {
    if (!this.projection) throw new Error('Study Score is not loaded.');
    assertRestoredReplayDecisionBinds(this.projection, restored);
    if (this.selectedDecision?.address !== restored.address) {
      throw new Error('Returned Study decision differs from the Score selection.');
    }
    this.restored = cloneDto(restored);
    this.phase = 'score';
    this.attemptId = null;
    this.branchObservation = null;
    this.plans = [];
    this.selectedPlan = null;
    this.preview = null;
    this.errorMessage = null;
  }

  resetSelection(): void {
    this.selectedDecision = null;
    this.restored = null;
    this.phase = 'score';
    this.attemptId = null;
    this.branchObservation = null;
    this.plans = [];
    this.selectedPlan = null;
    this.preview = null;
    this.busy = false;
    this.errorMessage = null;
  }
}

export function createStudyStore(): StudyStore {
  return new StudyStore();
}
