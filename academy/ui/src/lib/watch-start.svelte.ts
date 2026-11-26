import type { WorkStatus } from './types';

type Created = {id: string; finished: boolean};
type SetupResult = {work: WorkStatus | null; created: Created | null; error: string | null};
type Envelope = {handle: string; request: Record<string, unknown>};
const pendingKey = 'academy.watch-setup';

/** One unresolved tab-owned start. Only an explicit new start can load bots. */
export class WatchStart {
  work = $state<WorkStatus | null>(null);
  checking = $state(false);
  private controller: AbortController | null = null;
  private pending: Promise<Created> | null = null;

  start(request: Record<string, unknown>): Promise<Created> {
    if (this.pending) return this.pending;
    const pending = this.run(request);
    this.pending = pending;
    void pending.finally(() => { if (this.pending === pending) this.pending = null; }).catch(() => {});
    return pending;
  }

  cancel(): void { this.controller?.abort(); this.work = null; this.checking = false; }

  private async run(request: Record<string, unknown>): Promise<Created> {
    const saved = sessionStorage.getItem(pendingKey);
    const envelope: Envelope = saved ? JSON.parse(saved) : {handle: crypto.randomUUID(), request};
    // Persist before sending. A reload checks this handle, even if the current
    // form has changed. Never submit a changed configuration under its identity.
    sessionStorage.setItem(pendingKey, JSON.stringify(envelope));
    const controller = new AbortController();
    this.controller = controller;
    this.work = null; this.checking = true;
    const path = `/api/watch/setups/${encodeURIComponent(envelope.handle)}`;
    const read = async (init: RequestInit = {}): Promise<SetupResult> => {
      const response = await fetch(path, {...init, signal: controller.signal});
      const result = await response.json();
      if (!response.ok) {
        if (response.status === 404 && saved) sessionStorage.removeItem(pendingKey);
        throw new Error(typeof result.detail === 'string' ? result.detail : result.detail?.message ?? 'Watch setup is unavailable.');
      }
      return result as SetupResult;
    };
    // The first request owns creation. Every subsequent request is a read.
    const posting = saved ? null : read({method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(envelope.request)});
    // Observe rejection immediately even while a status read is outstanding.
    void posting?.catch(() => {});
    let timer: ReturnType<typeof setTimeout> | undefined;
    const wait = (): Promise<null> => new Promise((resolve, reject) => {
      const abort = () => { clearTimeout(timer); reject(new Error('Watch setup check stopped.')); };
      timer = setTimeout(() => { controller.signal.removeEventListener('abort', abort); resolve(null); }, 1000);
      if (controller.signal.aborted) abort();
      else controller.signal.addEventListener('abort', abort, {once: true});
    });
    try {
      for (;;) {
        const completed = posting ? await Promise.race([posting, wait()]) : null;
        clearTimeout(timer);
        const result = completed ?? await read();
        if (controller.signal.aborted) throw new Error('Watch setup check stopped.');
        const work = result.work;
        if (work?.scope === 'setup' && work.setup_id === envelope.handle && work.sequence >= (this.work?.sequence ?? 0)) this.work = work;
        this.checking = false;
        if (result.error) { sessionStorage.removeItem(pendingKey); throw new Error(result.error); }
        if (result.created) {
          sessionStorage.removeItem(pendingKey);
          return result.created;
        }
        // Reconciliation reads have the same cadence as the initial request.
        if (saved || completed) await wait();
      }
    } catch (reason) {
      if (!controller.signal.aborted) this.checking = sessionStorage.getItem(pendingKey) !== null;
      throw reason;
    } finally {
      controller.abort();
      clearTimeout(timer);
      await posting?.catch(() => {});
      this.work = null;
      if (this.controller === controller) this.controller = null;
    }
  }
}
export const watchStart = new WatchStart();
