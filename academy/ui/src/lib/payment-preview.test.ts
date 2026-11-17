import { describe, expect, it } from 'vitest';
import fixture from '../../../../protocol/fixtures/bolt-target.json';
import { PaymentPreviewController, type PreviewAddress, type PreviewResponse, type PreviewState } from './payment-preview';
import type { ExperienceFrame } from './types';

const exact = { scope: 'spell_cost' as const, state: 'exact' as const, explanation: 'Exact', cost: '{U}', sources: [], pool_after_production: { amounts: [0, 1, 0, 0, 0, 0], classes: [] }, pool_after_spending: null };
function frame(): ExperienceFrame { return structuredClone(fixture.recovery.frame) as ExperienceFrame; }
function deferred(): { promise: Promise<PreviewResponse>; resolve: (response: PreviewResponse) => void } {
  let resolve!: (response: PreviewResponse) => void;
  const promise = new Promise<PreviewResponse>(done => resolve = done);
  return { promise, resolve };
}

describe('frame-scoped payment proposals', () => {
  it('ignores a delayed response after commitment or disconnect', async () => {
    const response = deferred();
    let address!: PreviewAddress;
    let visible: PreviewState | null = null;
    const controller = new PaymentPreviewController(async request => { address = request; return response.promise; }, state => visible = state);
    const current = frame();
    const pending = controller.select(current, current.offers[0].id, 'Cast');
    controller.invalidate();
    response.resolve({ ...address, viewer: current.prompt!.actor, preview: exact });
    await pending;
    expect(visible).toBeNull();
  });
  it('keeps the newly selected offer when an older response arrives last', async () => {
    const response = deferred();
    const current = frame();
    const [first, second] = current.offers;
    let oldAddress!: PreviewAddress;
    const controller = new PaymentPreviewController(async request => {
      if (request.offer_id === first.id) {
        oldAddress = request;
        return response.promise;
      }
      return { ...request, viewer: current.prompt!.actor, preview: exact };
    }, () => {});
    const pending = controller.select(current, first.id, first.label);
    await controller.select(current, second.id, second.label);
    response.resolve({ ...oldAddress, viewer: current.prompt!.actor,
      preview: { ...exact, state: 'unavailable', explanation: 'Old offer' } });
    await pending;
    expect(controller.state?.offer).toBe(second.id);
    expect(controller.state?.preview?.state).toBe('exact');
  });
  it('uses one request per action per frame and clears cached source identity at a new frame', async () => {
    let requests = 0;
    const controller = new PaymentPreviewController(async request => { requests++; return { ...request, viewer: current.prompt!.actor, preview: exact }; }, () => {});
    const current = frame();
    await controller.select(current, current.offers[0].id, 'Cast');
    controller.clear();
    await controller.select(current, current.offers[0].id, 'Cast');
    expect(requests).toBe(1);
    await controller.select({ ...current, revision: current.revision + 1 }, current.offers[0].id, 'Cast');
    expect(requests).toBe(2);
  });
  it('rejects crossed identities and allows explicit retry without hiding legal choices', async () => {
    let wrong = true;
    const current = frame();
    const controller = new PaymentPreviewController(async request => ({ ...request, frame_hash: wrong ? 'crossed' : request.frame_hash, viewer: current.prompt!.actor, preview: exact }), () => {});
    await controller.select(current, current.offers[0].id, 'Cast');
    expect(controller.state?.error).toBe(true);
    expect(controller.state?.preview).toBeNull();
    wrong = false;
    await controller.select(current, current.offers[0].id, 'Cast');
    expect(controller.state?.preview?.state).toBe('exact');
  });
});
