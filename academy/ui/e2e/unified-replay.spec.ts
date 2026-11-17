import { expect, test } from '@playwright/test';

// Live production and archived terminal playback share the replay controls.
test('live Watch resumes the same timeline after Study and reload', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({width: 1280, height: 800});
  await page.emulateMedia({reducedMotion: 'reduce', colorScheme: 'light'});
  await page.goto('/replay?watch=1');
  await page.getByRole('combobox', {name: 'Deal mode', exact: true}).selectOption('repeat');
  await page.getByLabel('Seed', {exact: true}).fill('0');
  await page.getByRole('combobox', {name: 'Seat 2 bot', exact: true}).selectOption('random');
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

  await page.reload();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  expect(new URL(page.url()).searchParams.get('position')).toBe(recordedKey);
  expect(produced).toHaveLength(0);
  expect(errors).toEqual([]);
});

test('verified evaluation import reads the same timeline without producing a policy decision', async ({page}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  test.setTimeout(120_000);
  // Terminal playback needs a completed recording, not a randomly sized
  // native game produced through hundreds of browser requests.
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
  await page.reload();
  await expect(slider).toHaveValue(String(last));
  expect(steps).toHaveLength(0);
  const finalCard = page.getByTestId('game-board').locator('button[data-object-id]').first();
  await finalCard.click();
  await expect(page.getByRole('region', {name: 'Inspected card', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Close inspection', exact: true}).click();

  for (const viewport of [{width: 1280, height: 800}, {width: 1440, height: 900}]) {
    for (const colorScheme of ['light', 'dark'] as const) {
      await page.setViewportSize(viewport);
      await page.emulateMedia({colorScheme});
      await page.getByTestId('game-result-dialog').scrollIntoViewIfNeeded();
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
  expect(steps).toHaveLength(0);
  expect(errors).toEqual([]);
});
