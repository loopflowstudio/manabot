/** Ordinary release-SPA work. Old DOM-response baselines remain frozen separately. */
import { writeFile } from 'node:fs/promises';
import { expect, test, type Request } from '@playwright/test';

interface Sample {
  match: string; revision: number; input: number; sent: number | null;
  message: number | null; applied: number | null; dom: number | null;
  renderingOpportunity: number | null; terminal: boolean;
}
interface Probe { samples: Sample[]; frames: number[]; working: boolean; }
declare global { interface Window { academyTiming: Probe; } }

function distribution(values: number[]): { n: number; p50: number; p95: number; max: number; misses100ms: number } {
  const sorted = [...values].sort((a, b) => a - b);
  return { n: sorted.length, p50: sorted[Math.max(0, Math.ceil(sorted.length * .5) - 1)] ?? 0,
    p95: sorted[Math.max(0, Math.ceil(sorted.length * .95) - 1)] ?? 0,
    max: sorted.at(-1) ?? 0, misses100ms: sorted.filter(value => value > 100).length };
}

for (const [seed, width, height, motion] of [
  [0, 1440, 900, 'no-preference'], [1, 1280, 800, 'reduce'],
] as const) {
  test(`ordinary release timing seed ${seed}`, async ({ page, browser }, info) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width, height });
    await page.emulateMedia({ reducedMotion: motion });
    await page.addInitScript((gameSeed: number) => {
      const probe: Probe = { samples: [], frames: [], working: false };
      window.academyTiming = probe;
      const Native = window.WebSocket;
      class TimedSocket extends Native {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          this.addEventListener('message', event => {
            const arrived = performance.now();
            const data = JSON.parse(event.data);
            if ('work' in data) {
              probe.working = data.work?.state === 'running';
              if (!probe.working) window.dispatchEvent(new Event('academy-timing-settled'));
            }
            const frame = data.update?.frame ?? data.recovery?.frame;
            const sample = probe.samples.at(-1);
            if (!sample || sample.sent === null || sample.message !== null || !frame || frame.match_id !== sample.match || frame.revision <= sample.revision) return;
            sample.message = arrived;
            sample.terminal = frame.projection.game_over;
            window.dispatchEvent(new Event('academy-timing-authority'));
            // App message listeners run synchronously; this microtask bounds parse/store
            // work, including the probe's own parse. It is not server-only latency.
            queueMicrotask(() => { sample.applied = performance.now(); });
          });
        }
        override send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
          if (typeof data === 'string') {
            const message = JSON.parse(data);
            // Rewrite here: a routed second socket can reach this probe before the app.
            if (message.type === 'new_game') {
              message.config.seed = gameSeed;
              // The picker lists no standalone Search player; the timed opponent is set on the request.
              if (gameSeed !== 0) { message.config.villain_type = 'search'; message.config.villain_sims = 64; }
              super.send(JSON.stringify(message));
              return;
            }
            if (message.type === 'command') {
              const sample = probe.samples.at(-1);
              if (sample) { sample.sent = performance.now(); sample.match = message.command.match_id; sample.revision = message.command.expected_revision; }
            }
          }
          super.send(data);
        }
      }
      window.WebSocket = TimedSocket;
      document.addEventListener('keydown', event => {
        if (event.key === 'Enter' && event.target instanceof Element && event.target.closest('[data-testid="action-option"]')) {
          probe.samples.push({ match: '', revision: -1, input: performance.now(), sent: null, message: null, applied: null, dom: null, renderingOpportunity: null, terminal: false });
        }
      }, true);
      new MutationObserver(() => {
        const sample = probe.samples.at(-1);
        const revision = Number(document.querySelector('main')?.getAttribute('data-frame-revision'));
        if (!sample || sample.message === null || sample.dom !== null || revision <= sample.revision) return;
        sample.dom = performance.now();
        requestAnimationFrame(() => requestAnimationFrame(() => { sample.renderingOpportunity = performance.now(); }));
      }).observe(document, { subtree: true, attributes: true, attributeFilter: ['data-frame-revision'] });
      let previous = 0;
      const frame = (now: number): void => { if (previous) probe.frames.push(now - previous); previous = now; requestAnimationFrame(frame); };
      requestAnimationFrame(frame);
    }, seed);
    const art: { url: string; ms: number; status: number | string; cacheControl?: string }[] = [];
    const requests = new Map<string, number>();
    page.on('request', request => { if (request.url().includes('/api/card-art/')) requests.set(request.url(), performance.now()); });
    page.on('response', response => {
      const start = requests.get(response.url());
      if (start !== undefined) art.push({ url: response.url(), ms: performance.now() - start, status: response.status(), cacheControl: response.headers()['cache-control'] });
    });
    page.on('requestfailed', request => {
      const start = requests.get(request.url());
      if (start !== undefined) art.push({ url: request.url(), ms: performance.now() - start, status: request.failure()?.errorText ?? 'unknown' });
    });
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Performance.enable');
    await cdp.send('Tracing.start', { categories: 'devtools.timeline', transferMode: 'ReturnAsStream' });
    await page.goto('/?setup=1');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByTestId('opponent-select').selectOption('random');
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    for (let input = 0; input < 30; input += 1) {
      // The receipt can render before following bot work finishes. Keep that
      // render sample, but wait for settled authority before choosing again.
      await page.evaluate(() => new Promise<void>(resolve => {
        if (!window.academyTiming.working) resolve();
        else window.addEventListener('academy-timing-settled', () => resolve(), {once: true});
      }));
      await expect.poll(async () => await page.getByTestId('action-option').count() > 0 || await page.getByTestId('game-result').isVisible()).toBe(true);
      if (await page.getByTestId('game-result').isVisible()) break;
      const actions = page.getByTestId('action-option');
      if (input === 15) await page.getByTestId('inspect-toggle').click();
      const labels = await actions.allTextContents();
      let pick = labels.findIndex(label => /Play (Island|Mountain)|Cast /.test(label));
      if (pick < 0) pick = input % labels.length;
      await expect(actions.nth(pick)).toBeEnabled();
      await actions.nth(pick).focus();
      await actions.nth(pick).press('Enter');
      // Search time is part of the authority metric, not a client render deadline.
      // Await the actual response under the existing test deadline before checking
      // rendering. CI's Search 64 response can exceed the assertion's 5s budget.
      await page.evaluate(() => new Promise<void>(resolve => {
        if (window.academyTiming.samples.at(-1)?.message != null) resolve();
        else window.addEventListener('academy-timing-authority', () => resolve(), { once: true });
      }));
      await expect.poll(() => page.evaluate(() => window.academyTiming.samples.at(-1)?.renderingOpportunity)).toBeTruthy();
    }
    const probe = await page.evaluate(() => window.academyTiming);
    const live = probe.samples.filter(sample => !sample.terminal && sample.renderingOpportunity !== null);
    expect(live.length).toBeGreaterThanOrEqual(15);
    const metrics = await cdp.send('Performance.getMetrics');
    const ended = new Promise<{ stream: string }>(resolve => cdp.once('Tracing.tracingComplete', resolve));
    await cdp.send('Tracing.end');
    const { stream } = await ended;
    let trace = '';
    while (true) { const chunk = await cdp.send('IO.read', { handle: stream }); trace += chunk.data; if (chunk.eof) break; }
    await cdp.send('IO.close', { handle: stream });
    await writeFile(info.outputPath('timeline.json'), trace);
    const events = (JSON.parse(trace) as { traceEvents: { name: string; dur?: number }[] }).traceEvents;
    const receipt = { recordedAt: new Date().toISOString(), stage: process.env.ACADEMY_TIMING_STAGE ?? 'after',
      browser: browser.version(), runtime: { node: process.version, platform: process.platform, arch: process.arch },
      artCacheState: process.env.ACADEMY_ART_CACHE_STATE ?? 'existing default server cache; not inventoried before launch', viewport: { width, height, dpr: 1, scale: await page.evaluate(() => visualViewport!.scale) },
      seed, opponent: seed === 0 ? 'random' : 'search-64', motion,
      definitions: { acknowledgement: 'input to socket send', response: 'input to first matching authority frame before parse; an accepted human receipt precedes following bot work. Includes network and work before that frame',
        applied: 'message to microtask after synchronous listeners (includes probe parse)', render: 'message to second rAF after matching DOM revision; rendering opportunity upper bound, not display latency',
        settling: 'wait for runner completion before the next input; excluded from this render sample',
        overhead: 'WebSocket wrapper (including seed injection), one extra JSON parse, authority event, mutation observer, rAF loop and CDP timeline; no Playwright WebSocket routing' },
      acknowledgement: distribution(live.map(s => s.sent! - s.input)), authority: distribution(live.map(s => s.message! - s.input)),
      apply: distribution(live.map(s => s.applied! - s.message!)), render: distribution(live.map(s => s.renderingOpportunity! - s.message!)),
      raf: distribution(probe.frames), longFrames50ms: probe.frames.filter(value => value > 50).length,
      terminal: probe.samples.filter(s => s.terminal), samples: probe.samples, art, metrics,
      timeline: Object.fromEntries(['Layout', 'UpdateLayoutTree', 'Paint', 'RunTask'].map(name => [name, { n: events.filter(e => e.name === name).length, totalMs: events.filter(e => e.name === name).reduce((sum, e) => sum + (e.dur ?? 0), 0) / 1000 }])) };
    await writeFile(info.outputPath('timing.json'), JSON.stringify(receipt, null, 2));
    await info.attach('timing.json', { body: JSON.stringify(receipt), contentType: 'application/json' });
    // Machine-dependent misses are reported, never hidden by a changed budget.
    expect(receipt.acknowledgement.n + receipt.terminal.length).toBe(probe.samples.length);
  });
}

// A fresh server art directory is supplied by the caller. Ordinary opening-hand
// requests only; no prefetch, hidden-name requests or repeated network workload.
test('cold visible-art delivery receipt', async ({ page, browser }, info) => {
  test.skip(process.env.ACADEMY_COLD_ART_PROOF !== '1', 'Supply a fresh isolated ACADEMY_CARD_ART_DIR for cold delivery');
  const started = new Map<Request, number>();
  const delivered: { url: string; ms: number; status: number | string }[] = [];
  page.on('request', request => { if (request.url().includes('/api/card-art/')) started.set(request, performance.now()); });
  page.on('requestfinished', async request => {
    const start = started.get(request);
    if (start !== undefined) delivered.push({ url: request.url(), ms: performance.now() - start, status: (await request.response())?.status() ?? 'unknown' });
  });
  page.on('requestfailed', request => {
    const start = started.get(request);
    if (start !== undefined) delivered.push({ url: request.url(), ms: performance.now() - start, status: request.failure()?.errorText ?? 'unknown' });
  });
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('action-option').first()).toBeVisible();
  await page.waitForLoadState('networkidle');
  await expect.poll(() => delivered.length, { timeout: 30_000 }).toBe(started.size);
  await writeFile(info.outputPath('cold-art.json'), JSON.stringify({ recordedAt: new Date().toISOString(), browser: browser.version(),
    directory: process.env.ACADEMY_CARD_ART_DIR, initiallyEmpty: true, definition: 'request start to full response body or request failure',
    delivered, success: distribution(delivered.filter(item => typeof item.status === 'number' && item.status === 200).map(item => item.ms)),
    cancellations: delivered.filter(item => item.status === 'net::ERR_ABORTED'),
    failures: delivered.filter(item => item.status !== 200 && item.status !== 'net::ERR_ABORTED'),
  }, null, 2));
  expect(started.size).toBeGreaterThan(0);
});
