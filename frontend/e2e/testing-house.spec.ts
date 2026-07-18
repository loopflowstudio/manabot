import { expect, test, type Browser, type BrowserContext, type Page, type TestInfo } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

interface SharedTable {
  pilotContext: BrowserContext;
  watcherContext: BrowserContext;
  pilot: Page;
  watcher: Page;
  tableId: string;
}

interface RenderedAuthority {
  frameRevision: number;
  updateSequence: number;
  log: string[];
  board: string;
}

interface ActionTiming {
  action: number;
  fromRevision: number;
  toRevision: number;
  latencyMs: number;
  maximumBroadcastLag: number;
}

interface RevisionAdvance {
  revision: number;
  updateSequence: number;
  observedAt: number;
}

const PINNED_ADVISOR = 'conditional-determinized-puct-v1';
const PINNED_COMPUTE = '2w-16s-paired-seed-197';

async function openSharedTable(
  browser: Browser,
  reducedMotion: 'no-preference' | 'reduce' = 'no-preference',
): Promise<SharedTable> {
  const pilotContext = await browser.newContext({
    permissions: ['clipboard-read', 'clipboard-write'],
    reducedMotion,
  });
  const pilot = await pilotContext.newPage();
  await pilot.goto('/');
  await expect(pilot.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await pilot.getByTestId('opponent-select').selectOption('passive');
  await pilot.getByRole('button', { name: 'New Game' }).first().click();

  const pilotPanel = pilot.getByTestId('testing-house-panel');
  await expect(pilotPanel).toContainText('You are the pilot');
  const tableId = await pilotPanel.getAttribute('data-table-id');
  expect(tableId).toBeTruthy();

  await pilot.getByTestId('copy-watcher-invite').click();
  const invite = await pilot.evaluate(() => navigator.clipboard.readText());
  expect(invite).toContain('#table=');
  expect(invite).toContain('&watch=');

  const watcherContext = await browser.newContext({ reducedMotion });
  const watcher = await watcherContext.newPage();
  await watcher.goto(invite);
  await expect(watcher.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await expect(watcher.getByTestId('testing-house-panel')).toContainText(
    'You are the watcher',
  );
  await expect(pilotPanel).toContainText('both connected');
  await expect(watcher.getByTestId('testing-house-panel')).toHaveAttribute(
    'data-table-id',
    tableId!,
  );
  return { pilotContext, watcherContext, pilot, watcher, tableId: tableId! };
}

async function closeSharedTable(table: SharedTable): Promise<void> {
  await table.watcherContext.close();
  await table.pilotContext.close();
}

async function normalizedBoardText(page: Page): Promise<string> {
  return (await page.getByTestId('game-board').innerText()).replace(/\s+/g, ' ').trim();
}

async function renderedAuthority(page: Page): Promise<RenderedAuthority> {
  const main = page.locator('main');
  return {
    frameRevision: Number(await main.getAttribute('data-frame-revision')),
    updateSequence: Number(await main.getAttribute('data-update-seq')),
    log: await page.getByTestId('log-entry').allTextContents(),
    board: await normalizedBoardText(page),
  };
}

async function expectNoAuthorityMutation(
  watcher: Page,
  attempt: () => Promise<void>,
): Promise<void> {
  const before = await renderedAuthority(watcher);
  await attempt();
  await watcher.waitForTimeout(100);
  expect(await renderedAuthority(watcher)).toEqual(before);
}

async function expectPoliteAnnouncement(page: Page, text: string): Promise<void> {
  const announcement = page.getByTestId('testing-house-announcement');
  await expect(announcement).toHaveAttribute('role', 'status');
  await expect(announcement).toHaveAttribute('aria-live', 'polite');
  await expect(announcement).toHaveAttribute('aria-atomic', 'true');
  await expect(announcement).toHaveText(text);
}

async function assertPinnedAdvice(page: Page): Promise<void> {
  const advice = page.getByTestId('decision-advice');
  await expect(advice).toBeVisible();
  if ((await advice.getAttribute('open')) === null) {
    await advice.locator('summary').click();
  }
  await expect(page.getByTestId('advice-footer')).toContainText(PINNED_ADVISOR);
  await expect(page.getByTestId('advice-footer')).toContainText(PINNED_COMPUTE);
  await expect(page.getByTestId('advice-deltas')).toBeVisible();
  await expect(page.getByTestId('advice-delta-row').first()).toContainText('policy');
}

async function synchronizedAction(
  pilot: Page,
  watcher: Page,
  action: number,
): Promise<ActionTiming> {
  const pilotMain = pilot.locator('main');
  const watcherMain = watcher.locator('main');
  const fromRevision = Number(await pilotMain.getAttribute('data-frame-revision'));
  const initialPilotSequence = Number(await pilotMain.getAttribute('data-update-seq'));
  const initialWatcherSequence = Number(await watcherMain.getAttribute('data-update-seq'));
  expect(Number(await watcherMain.getAttribute('data-frame-revision'))).toBe(fromRevision);

  const installAdvanceProbe = async (page: Page): Promise<void> => {
    await page.locator('main').evaluate((node, expectedRevision) => {
      const main = node as HTMLElement;
      const target = window as typeof window & {
        __testingHouseAdvance?: Promise<RevisionAdvance>;
      };
      target.__testingHouseAdvance = new Promise<RevisionAdvance>((resolve, reject) => {
        const capture = (): boolean => {
          const revision = Number(main.dataset.frameRevision);
          if (revision <= expectedRevision) return false;
          resolve({
            revision,
            updateSequence: Number(main.dataset.updateSeq),
            observedAt: Date.now(),
          });
          return true;
        };
        if (capture()) return;
        const observer = new MutationObserver(() => {
          if (!capture()) return;
          observer.disconnect();
          window.clearTimeout(timeout);
        });
        observer.observe(main, {
          attributes: true,
          attributeFilter: ['data-frame-revision'],
        });
        const timeout = window.setTimeout(() => {
          observer.disconnect();
          reject(new Error(`frame did not advance beyond ${expectedRevision}`));
        }, 15_000);
      });
    }, fromRevision);
  };
  await Promise.all([installAdvanceProbe(pilot), installAdvanceProbe(watcher)]);

  const startedAt = Date.now();
  await expect(pilot.getByTestId('action-option').first()).toBeEnabled();
  await pilot.getByTestId('action-option').first().click();
  const readAdvance = (page: Page): Promise<RevisionAdvance> => page.evaluate(async () => {
    const source = window as typeof window & {
      __testingHouseAdvance?: Promise<RevisionAdvance>;
    };
    if (!source.__testingHouseAdvance) throw new Error('revision probe was not installed');
    return source.__testingHouseAdvance;
  });
  const [pilotAdvance, watcherAdvance] = await Promise.all([
    readAdvance(pilot),
    readAdvance(watcher),
  ]);
  expect(watcherAdvance.revision).toBe(pilotAdvance.revision);
  const pilotAdvances = pilotAdvance.updateSequence - initialPilotSequence;
  const watcherAdvances = watcherAdvance.updateSequence - initialWatcherSequence;
  expect(pilotAdvances).toBe(1);
  expect(watcherAdvances).toBe(1);
  const maximumBroadcastLag = Math.max(pilotAdvances, watcherAdvances);
  expect(maximumBroadcastLag).toBeLessThanOrEqual(1);
  return {
    action,
    fromRevision,
    toRevision: pilotAdvance.revision,
    latencyMs: Math.max(pilotAdvance.observedAt, watcherAdvance.observedAt) - startedAt,
    maximumBroadcastLag,
  };
}

function p95(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil(sorted.length * 0.95) - 1];
}

async function finishMatch(pilot: Page, watcher: Page): Promise<void> {
  const gameOver = pilot.getByText('Game Over', { exact: true });
  for (let action = 0; action < 1_000; action += 1) {
    if (await gameOver.isVisible()) return;
    const firstAction = pilot.getByTestId('action-option').first();
    await expect(firstAction.or(gameOver)).toBeVisible({ timeout: 30_000 });
    if (await gameOver.isVisible()) return;
    await synchronizedAction(pilot, watcher, action + 21);
  }
  throw new Error('shared match did not reach terminal');
}

async function decisionAddresses(page: Page): Promise<string[]> {
  return page.getByTestId('restore-table-decision').evaluateAll((buttons) =>
    buttons.map((button) => button.getAttribute('data-decision-address') ?? ''),
  );
}

test('pilot and watcher prove shared authority through terminal Study', async ({ browser }, testInfo: TestInfo) => {
  test.setTimeout(180_000);
  const table = await openSharedTable(browser);
  const { pilot, watcher, tableId } = table;
  try {
    await expect(watcher.getByTestId('action-option').first()).toBeDisabled();
    await expect(watcher.getByRole('button', { name: 'Pilot only' })).toBeDisabled();
    expect(await normalizedBoardText(watcher)).toBe(await normalizedBoardText(pilot));

    const watcherAction = watcher.getByTestId('action-option').first();
    await expectNoAuthorityMutation(watcher, () => watcherAction.click({ force: true }));
    await expectNoAuthorityMutation(watcher, () => watcherAction.press('Enter'));
    await expectNoAuthorityMutation(watcher, () => watcherAction.press('Space'));
    await expectNoAuthorityMutation(watcher, () => watcher.keyboard.press('F6'));
    await expect(pilot.getByTestId('action-option').first()).toBeEnabled();

    const connection = watcher.getByTestId('connection-badge');
    await expect(connection).toHaveAttribute('role', 'status');
    await expect(connection).toHaveAttribute('aria-live', 'polite');
    await expect(connection).toHaveAttribute('aria-label', 'Connection status: connected');

    await expect(watcher.getByTestId('author-belief')).toBeEnabled();
    await watcher.getByTestId('author-belief').focus();
    await watcher.keyboard.press('Enter');
    await expect(watcher.getByTestId('testing-house-panel')).toContainText(
      'Your read · personal',
    );
    await expect(pilot.getByTestId('testing-house-panel')).not.toContainText('Shared read');
    await expectPoliteAnnouncement(watcher, 'Your read is private.');
    await watcher.getByRole('button', { name: /Compare advice for/ }).click();
    await assertPinnedAdvice(watcher);

    await watcher.getByTestId('share-belief').focus();
    await watcher.keyboard.press('Space');
    await expect(pilot.getByTestId('testing-house-panel')).toContainText(
      'Shared read · table',
    );
    await expectPoliteAnnouncement(pilot, 'A read was shared with the table.');
    await expectPoliteAnnouncement(watcher, 'A read was shared with the table.');
    await pilot.getByRole('button', { name: /Compare advice for/ }).click();
    await assertPinnedAdvice(pilot);

    const timings: ActionTiming[] = [];
    for (let action = 1; action <= 20; action += 1) {
      timings.push(await synchronizedAction(pilot, watcher, action));
    }
    const receipt = {
      tableId,
      samples: timings.length,
      p95Ms: p95(timings.map(({ latencyMs }) => latencyMs)),
      maximumBroadcastLag: Math.max(...timings.map(({ maximumBroadcastLag }) => maximumBroadcastLag)),
      timings,
    };
    await testInfo.attach('testing-house-20-action-latency.json', {
      body: Buffer.from(`${JSON.stringify(receipt, null, 2)}\n`),
      contentType: 'application/json',
    });
    console.info(
      `testing-house 20-action receipt: p95=${receipt.p95Ms.toFixed(1)}ms, maximum broadcast lag=${receipt.maximumBroadcastLag}`,
    );
    expect(receipt.samples).toBe(20);
    expect(receipt.p95Ms).toBeLessThanOrEqual(250);
    expect(receipt.maximumBroadcastLag).toBeLessThanOrEqual(1);

    const firstDecision = watcher.getByTestId('restore-table-decision').first();
    await expect(firstDecision).toBeVisible();
    await firstDecision.click();
    await expectPoliteAnnouncement(watcher, 'Decision 1 restored for isolated Study.');
    await watcher.getByTestId('retry-table-decision').filter({ hasText: 'Pass priority' }).click();
    await expect(watcher.getByTestId('participant-study-controls')).toContainText(
      'Branch board shown only to you',
    );
    await expectPoliteAnnouncement(
      watcher,
      'Isolated line updated. The live table was not changed.',
    );
    await expect(pilot.getByTestId('participant-study-controls')).toHaveCount(0);

    await synchronizedAction(pilot, watcher, 21);
    await expect(watcher.getByTestId('participant-study-controls')).toContainText(
      'Branch board shown only to you',
    );
    await watcher.getByTestId('return-from-participant-branch').click();
    await expectPoliteAnnouncement(watcher, 'Returned to recorded decision 1.');
    await watcher.getByTestId('return-to-live-table').click();
    await expect(watcher.getByTestId('participant-study-controls')).toHaveCount(0);
    await expectPoliteAnnouncement(watcher, 'Returned to the live table.');

    await pilot.getByTestId('transfer-pilot').focus();
    await pilot.keyboard.press('Enter');
    await expect(pilot.getByTestId('testing-house-panel')).toContainText('You are the watcher');
    await expect(watcher.getByTestId('testing-house-panel')).toContainText('You are the pilot');
    await expectPoliteAnnouncement(pilot, 'You are now the watcher.');
    await expectPoliteAnnouncement(watcher, 'You are now the pilot.');
    await expect(pilot.getByTestId('action-option').first()).toBeDisabled();
    await expect(watcher.getByTestId('action-option').first()).toBeEnabled();

    await watcher.setViewportSize({ width: 390, height: 844 });
    expect(await watcher.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    )).toBe(false);
    for (const button of await watcher.getByTestId('testing-house-panel').getByRole('button').all()) {
      const box = await button.boundingBox();
      if (box) expect(box.height).toBeGreaterThanOrEqual(44);
    }
    const accessibility = await new AxeBuilder({ page: watcher })
      .include('[data-testid="testing-house-panel"]')
      .analyze();
    expect(accessibility.violations).toEqual([]);

    await pilot.reload();
    await expect(pilot.getByTestId('connection-badge')).toHaveText('connected', {
      timeout: 15_000,
    });
    await expect(pilot.getByTestId('connection-badge')).toHaveAttribute(
      'aria-label',
      'Connection status: connected',
    );
    await expect(pilot.getByTestId('testing-house-panel')).toContainText('You are the watcher');
    await expect(watcher.getByTestId('testing-house-panel')).toContainText('You are the pilot');
    await expect(pilot.getByTestId('testing-house-panel')).toHaveAttribute('data-table-id', tableId);

    await finishMatch(watcher, pilot);
    const pilotPanel = pilot.getByTestId('testing-house-panel');
    const watcherPanel = watcher.getByTestId('testing-house-panel');
    await expect(pilotPanel).toHaveAttribute('data-table-id', tableId);
    await expect(watcherPanel).toHaveAttribute('data-table-id', tableId);
    await expect(pilotPanel).toHaveAttribute('data-table-mode', 'study');
    await expect(watcherPanel).toHaveAttribute('data-table-mode', 'study');
    await expect(pilotPanel).toContainText('Testing house · study');
    await expect(watcherPanel).toContainText('Testing house · study');
    const pilotDecisions = await decisionAddresses(pilot);
    const watcherDecisions = await decisionAddresses(watcher);
    expect(pilotDecisions.length).toBeGreaterThan(0);
    expect(watcherDecisions).toEqual(pilotDecisions);
  } finally {
    await closeSharedTable(table);
  }
});

test('reduced motion preserves the shared-role read and branch semantics', async ({ browser }) => {
  const table = await openSharedTable(browser, 'reduce');
  const { pilot, watcher } = table;
  try {
    await expect(pilot.getByTestId('decision-advice')).toHaveAttribute('data-reduced-motion', 'true');
    await expect(watcher.getByTestId('decision-advice')).toHaveAttribute('data-reduced-motion', 'true');
    expect(await normalizedBoardText(watcher)).toBe(await normalizedBoardText(pilot));

    await watcher.getByTestId('author-belief').click();
    await expectPoliteAnnouncement(watcher, 'Your read is private.');
    await watcher.getByTestId('share-belief').click();
    await expectPoliteAnnouncement(pilot, 'A read was shared with the table.');
    await pilot.getByRole('button', { name: /Compare advice for/ }).click();
    await assertPinnedAdvice(pilot);

    await synchronizedAction(pilot, watcher, 1);
    await watcher.getByTestId('restore-table-decision').first().click();
    await expectPoliteAnnouncement(watcher, 'Decision 1 restored for isolated Study.');
    await watcher.getByTestId('retry-table-decision').filter({ hasText: 'Pass priority' }).click();
    await expectPoliteAnnouncement(
      watcher,
      'Isolated line updated. The live table was not changed.',
    );
    await watcher.getByTestId('return-from-participant-branch').click();
    await expectPoliteAnnouncement(watcher, 'Returned to recorded decision 1.');
    await watcher.getByTestId('return-to-live-table').click();
    await expectPoliteAnnouncement(watcher, 'Returned to the live table.');
    expect(await normalizedBoardText(watcher)).toBe(await normalizedBoardText(pilot));
  } finally {
    await closeSharedTable(table);
  }
});
