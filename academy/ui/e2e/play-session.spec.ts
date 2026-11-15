import { expect, test, type Page } from '@playwright/test';

test.use({ actionTimeout: 10_000 });

async function completeShortGame(page: Page): Promise<void> {
  // Eight-card imported deck-out fixtures exercise complete native games. They
  // measure workflow continuity, never trained strength or human endurance.
  for (let decision = 0; decision < 100; decision++) {
    if (await page.getByRole('button', {name: 'Rematch', exact: true}).count() >= 2) return;
    const revision = await page.locator('main').getAttribute('data-update-seq');
    const offers = page.getByTestId('action-option');
    const pass = page.locator('[data-testid="action-option"][data-action-type="PRIORITY_PASS_PRIORITY"]');
    await expect(offers.first()).toBeEnabled();
    await (await pass.count() ? pass.first() : offers.first()).click();
    await expect(page.locator('main')).not.toHaveAttribute('data-update-seq', revision!);
  }
  throw new Error('Bounded native game did not complete');
}

for (const [theme, width, height] of [['light', 1280, 800], ['dark', 1440, 900]] as const) {
  test(`repeated imported model session (${theme}, ${width}x${height})`, async ({page, context}, testInfo) => {
    await page.setViewportSize({width, height});
    await page.addInitScript(theme => localStorage.setItem('etude.theme', theme), theme);
    await page.emulateMedia({colorScheme: theme, reducedMotion:'reduce'});
    const errors: string[] = [];
    const adviceRequests: string[] = [];
    page.on('request', request => {if (request.url().includes('/api/advice')) adviceRequests.push(request.url());});
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByText('Import decklist files', {exact:true}).click();
    await page.getByLabel('Use imported decks for the next game').check();
    await page.getByLabel('Your deck source', {exact:true}).fill('8 Island');
    await page.getByLabel('Opponent deck source', {exact:true}).fill('8 Mountain');
    await page.getByRole('button', {name:'Check decklists', exact:true}).click();
    await expect(page.getByText(/Ready — development-v1/)).toBeVisible();
    await page.getByTestId('opponent-select').selectOption('policy');
    const catalog = await (await page.request.get('/api/opponents')).json();
    const first = catalog.opponents.find((p: {name:string}) => p.name === 'Session smoke A');
    const other = catalog.opponents.find((p: {name:string}) => p.name === 'Session smoke B');
    await page.getByTestId('checkpoint-select').selectOption(`policy:${first.sha256}`);
    await page.getByRole('combobox', {name:'Deal mode', exact:true}).selectOption('repeat');
    await page.getByLabel('Seed', {exact:true}).fill('18446744073709551615');
    await page.getByRole('button', {name:'New Game', exact:true}).click();
    const strip = page.getByRole('region', {name:'Play session', exact:true});
    await expect(strip).toContainText('Game 1');
    await completeShortGame(page);
    await strip.getByRole('button', {name:'Rematch', exact:true}).focus();
    await page.keyboard.press('Enter');
    await expect(strip).toContainText('Game 2');
    await completeShortGame(page);
    await strip.getByRole('button', {name:'Swap sides', exact:true}).focus();
    await page.keyboard.press('Enter');
    await expect(strip).toContainText('Game 3 · Draw first');
    await page.reload();
    await expect(strip).toContainText('Game 3 · Draw first');
    await expect(strip).toContainText('18446744073709551615');
    await completeShortGame(page);
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({path:testInfo.outputPath(`session-${theme}.png`)});
    const history = await strip.getByRole('link', {name:'Session history'}).getAttribute('href');
    await strip.getByRole('button', {name:'Change opponent', exact:true}).click();
    await page.getByTestId('checkpoint-select').selectOption(`policy:${other.sha256}`);
    await page.getByRole('button', {name:'Start with opponent', exact:true}).click();
    await expect(strip).toContainText('Game 4 · Draw first');
    await strip.getByText('Exact game configuration', {exact:true}).click();
    await expect(strip).toContainText(other.sha256);
    await expect(strip).toContainText('8 Island');
    await page.route('**/api/traces?*', async route => { const url = new URL(route.request().url()); if (url.searchParams.has('session')) url.searchParams.set('limit','2'); await route.continue({url:url.href}); });
    await page.goto(history!);
    await expect(page.getByRole('region', {name:'Session results'})).toContainText('out of 2 completed');
    await expect(page.getByTestId('game-history-row')).toHaveCount(2);
    await page.getByRole('link',{name:'Older games',exact:true}).click();
    await expect(page.getByRole('region',{name:'Session results'})).toContainText('out of 2 completed');
    const replay = page.locator('a[aria-label^="Replay game"]').last();
    // Use the archive's existing recording links and safe return route.
    await expect(replay).toBeVisible();
    await replay.click();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    await page.goto(history!);
    await page.getByRole('link', {name:'Return to current Play session'}).click();
    await expect(strip).toContainText('Game 4 · Draw first');
    await page.reload();
    await expect(strip).toContainText('Game 4 · Draw first');
    const independent = await context.newPage();
    await independent.goto('/');
    await expect(independent.getByRole('region', {name:'Play session'})).toHaveCount(0);
    await independent.close();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.getByRole('link', {name:'New game', exact:true}).first().click();
    await page.getByRole('button', {name:'New Game', exact:true}).click();
    await expect(strip).toContainText('Game 1');
    await page.goto(history!);
    await expect(page.getByRole('region', {name:'Session results'})).toContainText('1 stopped');
    await page.screenshot({path:testInfo.outputPath(`history-${theme}.png`)});
    expect(errors).toEqual([]);
    expect(adviceRequests).toEqual([]);
  });
}

test('one seat-oriented setup starts both registered bots and retains their exact setup', async ({page}, testInfo) => {
  await page.setViewportSize({width:1280,height:800});
  await page.goto('/?setup=1');
  await page.getByRole('combobox', {name:'Seat 1 controller',exact:true}).selectOption('bot');
  const catalog = await (await page.request.get('/api/opponents')).json();
  const smoke = catalog.opponents.filter((p: {name:string}) => p.name.startsWith('Session smoke'));
  for (const seat of [0,1]) {
    await page.getByTestId('opponent-select').nth(seat).selectOption('policy');
    await page.getByTestId('checkpoint-select').nth(seat).selectOption(`policy:${smoke[seat].sha256}`);
  }
  await page.getByText('Import decklist files',{exact:true}).click();
  await page.getByLabel('Use imported decks for the next game').check();
  await page.getByLabel('Seat 1 deck source',{exact:true}).fill('8 Island');
  await page.getByLabel('Seat 2 deck source',{exact:true}).fill('8 Mountain');
  await page.getByRole('button',{name:'Start watching',exact:true}).click();
  await expect(page.getByTestId('position-counter')).toBeVisible();
  await page.getByRole('button',{name:'Continue',exact:true}).click();
  await expect(page.getByTestId('recorded-decision-evidence')).toContainText('Recorded at this decision');
  await expect(page.getByRole('link',{name:'Current game',exact:true}).first()).toHaveAttribute('href',/trace=/);
  await page.reload();
  await expect(page.getByTestId('position-counter')).toBeVisible();
  await page.getByText('Watch another matchup',{exact:true}).click();
  await expect(page.getByTestId('checkpoint-select').nth(0)).toHaveValue(`policy:${smoke[0].sha256}`);
  await expect(page.getByTestId('checkpoint-select').nth(1)).toHaveValue(`policy:${smoke[1].sha256}`);
  await expect(page.getByLabel('Seat 1 deck source',{exact:true})).toHaveValue('8 Island');
  await page.screenshot({path:testInfo.outputPath('shared-watch-setup.png')});
  await page.getByRole('combobox',{name:'Seat 2 controller',exact:true}).selectOption('human');
  await expect(page.getByLabel('Your deck source',{exact:true})).toHaveValue('8 Mountain');
  await expect(page.getByLabel('Opponent deck source',{exact:true})).toHaveValue('8 Island');
});

test('session storage failure leaves current Play usable with an honest warning', async ({page}) => {
  await page.addInitScript(() => Object.defineProperty(window,'sessionStorage',{get:() => {throw new DOMException('Disabled','SecurityError');}}));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button',{name:'New Game',exact:true}).click();
  await expect(page.getByText(/Reload continuity is unavailable/)).toBeVisible();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  const before = await page.locator('main').getAttribute('data-frame-revision');
  await page.getByTestId('action-option').first().click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision',before!);
});
