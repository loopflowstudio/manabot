import { expect, test } from '@playwright/test';

// One bounded real native game. This is workflow evidence, not a strength test.
test('live Watch, archived playback and the final action share one resumable timeline', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1280, height: 800});
  await page.emulateMedia({reducedMotion: 'reduce', colorScheme: 'light'});
  await page.goto('/replay?watch=1');
  await page.getByRole('combobox', {name: 'Seat 2', exact: true}).selectOption('random');
  await page.getByRole('button', {name: 'Start watching', exact: true}).click();
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  await expect(page.getByRole('checkbox', {name: 'Show both hands'})).not.toBeChecked();
  const id = new URL(page.url()).searchParams.get('trace')!;
  await expect.poll(() => page.evaluate(id => sessionStorage.getItem(`etude.watch.${id}`) !== null, id)).toBe(true);
  await expect.poll(() => page.evaluate(id => localStorage.getItem(`etude.watch.${id}`), id)).toBeNull();

  // Earlier reads cannot produce. Study retries have a separate rules authority.
  const produced: string[] = [];
  page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/step')) produced.push(request.url()); });
  const recordedKey = new URL(page.url()).searchParams.get('position');
  await page.getByRole('button', {name: 'Try another line', exact: true}).click();
  await page.getByTestId('action-option').first().click();
  await page.getByTestId('study-return').click();
  await expect(page.getByTestId('study-panel')).toHaveCount(0);
  expect(new URL(page.url()).searchParams.get('position')).toBe(recordedKey);
  expect(produced).toHaveLength(0);

  // Complete the same ordinary seeded game through its existing producer.
  const terminal = await page.evaluate(async id => {
    const token = sessionStorage.getItem(`etude.watch.${id}`)!;
    const headers = {'content-type': 'application/json', 'x-etude-participant-token': token};
    const indexResponse = await fetch(`/api/traces/${id}/recording`, {headers});
    const index = await indexResponse.json();
    let revision = index.revision as number;
    for (let decision = 0; decision < 350; decision++) {
      const response = await fetch(`/api/watch/${id}/step`, {method: 'POST', headers, body: JSON.stringify({revision})});
      if (!response.ok) throw new Error(`Native producer failed: ${await response.text()}`);
      const result = await response.json();
      revision = result.revision;
      if (result.finished) return {revision};
    }
    throw new Error('Seeded recording exceeded its bounded decision count');
  }, id);
  expect(terminal.revision).toBeGreaterThan(1);
  produced.length = 0;
  await page.reload();
  await expect(page.getByTestId('position-counter')).toContainText('Position 1 /');
  const slider = page.getByRole('slider', {name: 'Replay position'});
  const last = Number(await slider.getAttribute('max'));
  await slider.fill(String(last - 2));
  await expect(page.getByTestId('game-result-dialog')).toHaveCount(0);
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(page.getByTestId('game-result-dialog')).toBeVisible();
  await expect(page.getByTestId('replay-perspective')).toContainText('Recorded seat-0 view');
  const history = page.getByTestId('presentation-history');
  await history.locator('summary').click();
  await expect(history.getByRole('listitem').last()).toContainText('damage');
  const historyScope = await history.getAttribute('data-scope');
  await page.getByRole('button', {name: 'Swap seats'}).click();
  await expect(history).toHaveAttribute('data-scope', historyScope!);
  await page.reload();
  await expect(page.getByText('Near: seat 2', {exact: true})).toBeVisible();
  await expect(slider).toHaveValue(String(last - 1));
  await page.getByRole('button', {name: 'Next', exact: true}).focus();
  await page.keyboard.press('Enter');
  await expect(slider).toHaveValue(String(last));
  await expect(page.getByRole('button', {name: 'Next', exact: true})).toBeDisabled();
  expect(produced).toHaveLength(0);
  const finalCard = page.getByTestId('game-board').locator('button[data-object-id]').first();
  await finalCard.click();
  await expect(page.getByRole('region', {name: 'Inspected card', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Close inspection', exact: true}).click();

  for (const viewport of [{width: 1280, height: 800}, {width: 1440, height: 900}]) {
    for (const colorScheme of ['light', 'dark'] as const) {
      await page.setViewportSize(viewport);
      await page.emulateMedia({colorScheme});
      await expect(page.getByTestId('game-result-dialog')).toBeInViewport({ratio: 1});
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({path: testInfo.outputPath(`result-${viewport.width}-${colorScheme}.png`)});
    }
  }
  const markers = page.getByRole('combobox', {name: 'Jump to turn and phase'});
  await markers.selectOption({index: 1});
  await expect(slider).toHaveValue('0');
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveValue('1');
  await page.getByRole('combobox', {name: 'Speed', exact: true}).selectOption('16');
  await page.getByRole('button', {name: 'Play', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Pause', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  expect(produced).toHaveLength(0);
  expect(errors).toEqual([]);
});

test('verified evaluation import reads the same timeline without producing a policy decision', async ({page}) => {
  test.setTimeout(120_000);
  const sourceResponse = await page.request.get('/api/__fixtures/recorded-evaluation');
  expect(sourceResponse.ok()).toBe(true);
  const source = await sourceResponse.text();
  const steps: string[] = [];
  page.on('request', request => { if (request.url().endsWith('/step')) steps.push(request.url()); });
  await page.goto('/replay?watch=1');
  await page.getByLabel('Open a game file').setInputFiles({name: 'evaluation.json', mimeType: 'application/json', buffer: Buffer.from(source)});
  await expect(page.getByTestId('game-board')).toBeVisible({timeout: 60_000});
  await expect(page.getByRole('heading', {name: 'The Score · Evaluation game', exact: true})).toBeVisible();
  const slider = page.getByRole('slider', {name: 'Replay position'});
  const last = Number(await slider.getAttribute('max'));
  await slider.fill(String(last - 2));
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(page.getByTestId('game-result-dialog')).toBeVisible();
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(slider).toHaveValue(String(last));
  await page.reload();
  await expect(slider).toHaveValue(String(last));
  expect(steps).toEqual([]);
});
