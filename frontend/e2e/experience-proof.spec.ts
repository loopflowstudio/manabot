import { expect, test, type Browser, type Page } from '@playwright/test';

import { EXPERIENCE_PROOF_BASELINE, REACHABLE_PROMPT_FAMILIES } from './experience-proof-baseline';

const RESUME_STORAGE_KEY = 'etude.gui.resume';
const CALIBRATING = process.env.ETUDE_CALIBRATE_PROOF === '1';
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==',
  'base64',
);

interface Distribution {
  p50: number;
  p95: number;
  max: number;
}

interface InteractionProbe {
  acknowledgementMs: number | null;
  authorityResponseMs: number | null;
}

interface ProofState {
  frameDeltas: number[];
  sockets: WebSocket[];
  interaction: InteractionProbe | null;
}

interface ObservedProof {
  schemaVersion: number;
  recordedAt: string;
  browserVersion: string;
  scope: typeof EXPERIENCE_PROOF_BASELINE.scope;
  promptInventory: readonly string[];
  samples: typeof EXPERIENCE_PROOF_BASELINE.samples;
  metrics: {
    warmLaunchToPlayableMs: Distribution;
    inputAcknowledgementMs: Distribution;
    authorityResponseMs: Distribution;
    frameDeltaMs: Distribution & { longFrameCount: number };
    rendererHeapMiB: Distribution;
  };
}

type ProofWindow = Window & typeof globalThis & { __etudeProof?: ProofState };

function round(value: number, digits = 2): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function distribution(values: number[], digits = 2): Distribution {
  expect(values.length, 'measurement produced no samples').toBeGreaterThan(0);
  const sorted = [...values].sort((left, right) => left - right);
  const percentile = (fraction: number): number => {
    const index = Math.max(0, Math.ceil(sorted.length * fraction) - 1);
    return round(sorted[index], digits);
  };
  return {
    p50: percentile(0.5),
    p95: percentile(0.95),
    max: round(sorted[sorted.length - 1], digits),
  };
}

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

async function stubCardArt(page: Page): Promise<void> {
  await page.route('https://api.scryfall.com/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: TINY_PNG }),
  );
}

async function selectReferenceExperience(page: Page): Promise<void> {
  await page.getByTestId('deck-select-hero').selectOption(
    EXPERIENCE_PROOF_BASELINE.scope.heroDeck,
  );
  await page.getByTestId('deck-select-villain').selectOption(
    EXPERIENCE_PROOF_BASELINE.scope.villainDeck,
  );
  await page.getByTestId('opponent-select').selectOption(
    EXPERIENCE_PROOF_BASELINE.scope.opponent,
  );
}

async function startReferenceGame(page: Page): Promise<void> {
  await selectReferenceExperience(page);
  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('deck-names')).toHaveText('UR Lessons vs GW Allies', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 30_000 });
}

async function measureWarmLaunches(browser: Browser): Promise<number[]> {
  const measurements: number[] = [];
  for (let run = 0; run < EXPERIENCE_PROOF_BASELINE.samples.warmLaunches; run += 1) {
    const context = await browser.newContext();
    const page = await context.newPage();
    await stubCardArt(page);
    const startedAt = performance.now();
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
      timeout: 15_000,
    });
    await startReferenceGame(page);
    measurements.push(performance.now() - startedAt);
    await context.close();
  }
  return measurements;
}

async function installProofInstrumentation(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const proofWindow = window as ProofWindow;
    const nativeWebSocket = window.WebSocket;
    const sockets: WebSocket[] = [];

    class TrackedWebSocket extends nativeWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        if (protocols === undefined) {
          super(url);
        } else {
          super(url, protocols);
        }
        sockets.push(this);
      }
    }

    Object.defineProperty(window, 'WebSocket', {
      configurable: true,
      value: TrackedWebSocket,
      writable: true,
    });

    proofWindow.__etudeProof = {
      frameDeltas: [],
      sockets,
      interaction: null,
    };

    let previousFrame: number | null = null;
    const sampleFrame = (timestamp: number): void => {
      const proof = proofWindow.__etudeProof;
      if (proof && previousFrame !== null) {
        proof.frameDeltas.push(timestamp - previousFrame);
      }
      previousFrame = timestamp;
      window.requestAnimationFrame(sampleFrame);
    };
    window.requestAnimationFrame(sampleFrame);
  });
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function authoritativeBoardText(page: Page): Promise<string> {
  return page.getByTestId('game-board').evaluate((element) => {
    const board = element.cloneNode(true) as HTMLElement;
    board.querySelector('[data-testid="presentation-stage"]')?.remove();
    return (board.textContent ?? '').replace(/\s+/g, ' ').trim();
  });
}

async function armInteractionProbe(
  page: Page,
  initialLogCount: number,
  initialUpdateSequence: number,
): Promise<void> {
  await page.evaluate(
    ({ logCount, updateSequence }) => {
      const proofWindow = window as ProofWindow;
      const proof = proofWindow.__etudeProof;
      if (!proof) {
        throw new Error('experience proof instrumentation is unavailable');
      }
      proof.interaction = null;

      const onKeyDown = (event: KeyboardEvent): void => {
        const target = event.target;
        if (
          event.key !== 'Enter' ||
          !(target instanceof Element) ||
          target.closest('[data-testid="action-option"]') === null
        ) {
          return;
        }
        document.removeEventListener('keydown', onKeyDown, true);

        const startedAt = performance.now();
        const result: InteractionProbe = {
          acknowledgementMs: null,
          authorityResponseMs: null,
        };
        proof.interaction = result;

        const capture = (): void => {
          if (
            result.acknowledgementMs === null &&
            document.querySelectorAll('[data-testid="log-entry"]').length > logCount
          ) {
            result.acknowledgementMs = performance.now() - startedAt;
          }
          const currentSequence = Number(
            document.querySelector('main')?.getAttribute('data-update-seq') ?? '0',
          );
          if (result.authorityResponseMs === null && currentSequence > updateSequence) {
            result.authorityResponseMs = performance.now() - startedAt;
          }
          if (result.acknowledgementMs !== null && result.authorityResponseMs !== null) {
            observer.disconnect();
          }
        };

        const observer = new MutationObserver(capture);
        observer.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['data-update-seq'],
          childList: true,
          subtree: true,
        });
        capture();
      };

      document.addEventListener('keydown', onKeyDown, true);
    },
    { logCount: initialLogCount, updateSequence: initialUpdateSequence },
  );
}

async function chooseDeterministicAction(page: Page): Promise<number> {
  const labels = await page.getByTestId('action-option').allTextContents();
  const preferences = [
    /^Play /,
    /^Cast /,
    /^Activate /,
    /^Attack /,
    /^Block /,
    /^Target /,
    /^Keep /,
    /^Put /,
    /^Discard /,
    /^Pay /,
    /^Tap /,
    /^Decline/,
    /Pass priority/,
  ];
  for (const preference of preferences) {
    const index = labels.findIndex((label) => preference.test(label.trim()));
    if (index >= 0) {
      return index;
    }
  }
  return 0;
}

async function measureKeyboardAction(page: Page): Promise<InteractionProbe> {
  const actions = page.getByTestId('action-option');
  await expect(actions.first()).toBeVisible({ timeout: 30_000 });
  const actionIndex = await chooseDeterministicAction(page);
  const action = actions.nth(actionIndex);
  await expect(action).not.toHaveAccessibleName('');

  const initialLogCount = await page.getByTestId('log-entry').count();
  const initialUpdateSequence = await updateSequence(page);
  await action.focus();
  await expect(action).toBeFocused();
  await armInteractionProbe(page, initialLogCount, initialUpdateSequence);
  await action.press('Enter');

  await expect
    .poll(
      () =>
        page.evaluate(
          () =>
            (window as ProofWindow).__etudeProof?.interaction?.authorityResponseMs ?? null,
        ),
      { timeout: 30_000, message: 'authority did not respond to keyboard action' },
    )
    .not.toBeNull();

  const sample = await page.evaluate(
    () => (window as ProofWindow).__etudeProof?.interaction ?? null,
  );
  expect(sample?.acknowledgementMs, 'local input acknowledgement was not observed').not.toBeNull();
  expect(sample?.authorityResponseMs, 'authority response was not observed').not.toBeNull();
  return sample as InteractionProbe;
}

async function assertVisibleButtonsAreNamed(page: Page): Promise<void> {
  const buttons = page.getByRole('button');
  for (let index = 0; index < (await buttons.count()); index += 1) {
    const button = buttons.nth(index);
    if ((await button.isVisible()) && (await button.isEnabled())) {
      await expect(button, `visible enabled button ${index} has no accessible name`).not.toHaveAccessibleName(
        '',
      );
    }
  }
}

async function rendererHeapMiB(page: Page): Promise<number> {
  const session = await page.context().newCDPSession(page);
  await session.send('Performance.enable');
  const payload = (await session.send('Performance.getMetrics')) as {
    metrics?: Array<{ name: string; value: number }>;
  };
  await session.detach();
  const usedHeap = payload.metrics?.find((metric) => metric.name === 'JSHeapUsedSize')?.value;
  if (usedHeap === undefined) {
    throw new Error('Chromium did not expose JSHeapUsedSize');
  }
  return usedHeap / (1024 * 1024);
}

async function injectReconnectFault(page: Page): Promise<void> {
  const badge = page.getByTestId('connection-badge');
  const credentialsBefore = await page.evaluate((key) => sessionStorage.getItem(key), RESUME_STORAGE_KEY);
  const boardBefore = await authoritativeBoardText(page);
  const actionsBefore = await page.getByTestId('action-option').allTextContents();
  const logCountBefore = await page.getByTestId('log-entry').count();
  const updateSequenceBefore = await updateSequence(page);
  expect(credentialsBefore, 'resume credentials were not stored').not.toBeNull();

  await page.evaluate(() => {
    const socket = (window as ProofWindow).__etudeProof?.sockets.at(-1);
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error('no open WebSocket is available for reconnect fault injection');
    }
    socket.close(4101, 'experience proof reconnect fault');
  });

  await expect(badge).toHaveText('disconnected', { timeout: 5_000 });
  await expect(badge).toHaveText('connected', { timeout: 15_000 });
  await expect
    .poll(() => updateSequence(page), {
      timeout: 15_000,
      message: 'resume response did not advance the client update sequence',
    })
    .toBeGreaterThan(updateSequenceBefore);

  const credentialsAfter = await page.evaluate((key) => sessionStorage.getItem(key), RESUME_STORAGE_KEY);
  const boardAfter = await authoritativeBoardText(page);
  const actionsAfter = await page.getByTestId('action-option').allTextContents();
  const socketCount = await page.evaluate(
    () => (window as ProofWindow).__etudeProof?.sockets.length ?? 0,
  );

  expect(credentialsAfter).toBe(credentialsBefore);
  expect(boardAfter).toBe(boardBefore);
  expect(actionsAfter).toEqual(actionsBefore);
  expect(await page.getByTestId('log-entry').count()).toBe(logCountBefore);
  expect(socketCount).toBeGreaterThanOrEqual(2);
}

function assertBudgets(observed: ObservedProof): void {
  if (CALIBRATING) {
    return;
  }
  const budgets = EXPERIENCE_PROOF_BASELINE.metrics;
  expect(observed.browserVersion).toBe(EXPERIENCE_PROOF_BASELINE.referenceProfile.chromium);
  expect(observed.metrics.warmLaunchToPlayableMs.p95).toBeLessThanOrEqual(
    budgets.warmLaunchToPlayableMs.budgetP95,
  );
  expect(observed.metrics.inputAcknowledgementMs.p95).toBeLessThanOrEqual(
    budgets.inputAcknowledgementMs.budgetP95,
  );
  expect(observed.metrics.authorityResponseMs.p95).toBeLessThanOrEqual(
    budgets.authorityResponseMs.budgetP95,
  );
  expect(observed.metrics.frameDeltaMs.p95).toBeLessThanOrEqual(
    budgets.frameDeltaMs.budgetP95,
  );
  expect(observed.metrics.frameDeltaMs.max).toBeLessThanOrEqual(
    budgets.frameDeltaMs.budgetMax,
  );
  expect(observed.metrics.rendererHeapMiB.max).toBeLessThanOrEqual(
    budgets.rendererHeapMiB.budgetMax,
  );
}

test('reference experience meets baseline, resumes, and accepts keyboard play', async ({
  browser,
  page,
}, testInfo) => {
  test.setTimeout(300_000);
  const consoleErrors = collectConsoleErrors(page);
  const warmLaunches = await measureWarmLaunches(browser);

  await installProofInstrumentation(page);
  await stubCardArt(page);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  const heapSamples: number[] = [await rendererHeapMiB(page)];
  await startReferenceGame(page);
  await assertVisibleButtonsAreNamed(page);

  const inputAcknowledgements: number[] = [];
  const authorityResponses: number[] = [];
  heapSamples.push(await rendererHeapMiB(page));

  for (let interaction = 0; interaction < EXPERIENCE_PROOF_BASELINE.samples.interactions; interaction += 1) {
    if (interaction === Math.floor(EXPERIENCE_PROOF_BASELINE.samples.interactions / 2)) {
      await injectReconnectFault(page);
      heapSamples.push(await rendererHeapMiB(page));
    }

    const sample = await measureKeyboardAction(page);
    inputAcknowledgements.push(sample.acknowledgementMs as number);
    authorityResponses.push(sample.authorityResponseMs as number);
    heapSamples.push(await rendererHeapMiB(page));

    const gameOver = page.getByText('Game Over', { exact: true });
    if (await gameOver.isVisible()) {
      await page.getByRole('button', { name: 'Play Again' }).click();
      await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 30_000 });
    }
  }

  const frameDeltas = await page.evaluate(
    () => (window as ProofWindow).__etudeProof?.frameDeltas ?? [],
  );
  expect(await page.evaluate(() => document.visibilityState)).toBe('visible');
  expect(frameDeltas.length, 'too few animation frames were sampled').toBeGreaterThan(30);

  const frameDistribution = distribution(frameDeltas);
  const observed: ObservedProof = {
    schemaVersion: EXPERIENCE_PROOF_BASELINE.schemaVersion,
    recordedAt: new Date().toISOString(),
    browserVersion: browser.version(),
    scope: EXPERIENCE_PROOF_BASELINE.scope,
    promptInventory: REACHABLE_PROMPT_FAMILIES,
    samples: EXPERIENCE_PROOF_BASELINE.samples,
    metrics: {
      warmLaunchToPlayableMs: distribution(warmLaunches),
      inputAcknowledgementMs: distribution(inputAcknowledgements),
      authorityResponseMs: distribution(authorityResponses),
      frameDeltaMs: {
        ...frameDistribution,
        longFrameCount: frameDeltas.filter(
          (delta) => delta > EXPERIENCE_PROOF_BASELINE.metrics.frameDeltaMs.longFrameThresholdMs,
        ).length,
      },
      rendererHeapMiB: distribution(heapSamples, 3),
    },
  };

  await testInfo.attach('experience-proof.json', {
    body: Buffer.from(JSON.stringify(observed, null, 2)),
    contentType: 'application/json',
  });
  console.log(`experience proof: ${JSON.stringify(observed)}`);

  assertBudgets(observed);
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});
