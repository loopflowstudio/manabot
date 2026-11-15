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
    await expect(page.locator('[data-testid="action-option"]:enabled').first().or(page.getByText('Game Over', {exact: true}))).toBeVisible();
    if (await page.getByRole('button', {name: 'Rematch', exact: true}).count() >= 2) return;
    await (await pass.count() ? pass.first() : offers.first()).click();
    await expect(page.locator('main')).not.toHaveAttribute('data-update-seq', revision!);
  }
  throw new Error('Bounded native game did not complete');
}

for (const [theme, width, height] of [['light', 1280, 800], ['dark', 1440, 900]] as const) {
  test(`repeated imported model session (${theme}, ${width}x${height})`, async ({page, context, browser}, testInfo) => {
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
    await expect(page.getByRole('region', {name:'New session setup'})).toContainText('Choose your setup to start a new session.');
    await expect(page.getByRole('region', {name:'New session setup'})).not.toContainText('current session draft');
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
    await page.getByRole('button', {name:'Start game', exact:true}).click();
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
    await page.getByRole('button', {name:'Start next game', exact:true}).click();
    await expect(strip).toContainText('Game 4 · Draw first');
    await strip.getByText('Exact game configuration', {exact:true}).click();
    await expect(strip).toContainText(other.sha256);
    await expect(strip).toContainText('8 Island');
    await page.route('**/api/traces?*', async route => { const url = new URL(route.request().url()); if (url.searchParams.has('session')) url.searchParams.set('limit','2'); await route.continue({url:url.href}); });
    await page.goto(history!);
    await expect(page.getByRole('region', {name:'Session results'})).toContainText('out of 2 completed');
    // A copied link carries no owner access; the original tab still recovers.
    const visitor = await browser.newContext();
    try {
      const visitorPage = await visitor.newPage();
      await visitorPage.goto(new URL(history!, page.url()).href);
      await expect(visitorPage.getByRole('alert')).toContainText('This browser cannot access this session history.');
      await expect(visitorPage.getByRole('button', {name:'Try again', exact:true})).toHaveCount(0);
    } finally { await visitor.close(); }
    const summaries = page.getByRole('region', {name:'Session results'}).locator('summary');
    await expect(summaries.filter({hasText:first.name})).toHaveCount(2);
    await expect(summaries.filter({hasText:other.name})).toHaveCount(1);
    await expect(summaries.filter({hasText:other.name})).toContainText(other.sha256.slice(0,10));
    await expect(page.getByTestId('game-history-row')).toHaveCount(2);
    await page.getByRole('link',{name:'Older games',exact:true}).click();
    await expect(page.getByRole('region',{name:'Session results'})).toContainText('out of 2 completed');
    const replay = page.locator('a[aria-label^="Replay game"]').last();
    // Use the archive's existing recording links and safe return route.
    await expect(replay).toBeVisible();
    await replay.click();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    await page.goto(history!);
    await page.getByRole('link', {name:'Return to current game'}).click();
    await expect(strip).toContainText('Game 4 · Draw first');
    await page.reload();
    await expect(strip).toContainText('Game 4 · Draw first');
    const independent = await context.newPage();
    await independent.goto('/');
    await expect(independent.getByRole('region', {name:'Play session'})).toHaveCount(0);
    await independent.close();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.getByRole('link', {name:'New game', exact:true}).first().click();
    await page.getByRole('button', {name:'Start game', exact:true}).click();
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
  await page.getByRole('button',{name:'Start game',exact:true}).click();
  await expect(page.getByText(/Reload continuity is unavailable/)).toBeVisible();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  const before = await page.locator('main').getAttribute('data-frame-revision');
  await page.getByTestId('action-option').first().click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision',before!);
});

for (const [theme, width, height] of [['light',1280,800], ['dark',1440,900]] as const) {
  test(`distinct editor intents preserve a live session (${theme})`, async ({page}, testInfo) => {
    await page.setViewportSize({width,height});
    await page.addInitScript(theme => localStorage.setItem('etude.theme',theme),theme);
    await page.emulateMedia({colorScheme:theme,reducedMotion:'reduce'});
    page.on('dialog',dialog => dialog.accept());
    await page.goto('/');
    await page.getByTestId('opponent-select').selectOption('passive');
    await page.getByRole('combobox',{name:'Deal mode',exact:true}).selectOption('repeat');
    await page.getByLabel('Seed',{exact:true}).fill('18446744073709551615');
    await page.getByRole('button',{name:'Start game',exact:true}).click();
    const strip = page.getByRole('region',{name:'Play session',exact:true});
    await expect(strip).toContainText('Game 1');
    const history = await strip.getByRole('link',{name:'Session history'}).getAttribute('href');
    const revision = await page.locator('main').getAttribute('data-frame-revision');
    await strip.getByRole('button',{name:'Edit next game',exact:true}).focus();
    await page.keyboard.press('Enter');
    const next = page.getByRole('region',{name:'Next game setup',exact:true});
    await expect(next.getByRole('button',{name:'Start game',exact:true})).toHaveCount(0);
    await page.screenshot({path:testInfo.outputPath(`next-session-${theme}.png`)});
    await next.getByLabel('Seed',{exact:true}).fill('7');
    await next.getByRole('button',{name:'Save next setup',exact:true}).click();
    await expect(next).toContainText('Next-game setup saved');
    await expect(page.locator('main')).toHaveAttribute('data-frame-revision',revision!);
    await next.getByRole('button',{name:'Cancel',exact:true}).click();

    await page.getByRole('link',{name:'New game',exact:true}).first().click();
    const fresh = page.getByRole('region',{name:'New session setup',exact:true});
    await expect(fresh).toContainText('These edits do not change your current session draft.');
    await expect(fresh.getByRole('button',{name:'Save next setup',exact:true})).toHaveCount(0);
    await expect(fresh.getByRole('button',{name:'Start next game',exact:true})).toHaveCount(0);
    await page.screenshot({path:testInfo.outputPath(`new-session-${theme}.png`)});
    await fresh.getByLabel('Seed',{exact:true}).fill('9');
    await fresh.getByRole('button',{name:'Cancel',exact:true}).focus();
    await page.keyboard.press('Enter');
    await strip.getByRole('button',{name:'Edit next game',exact:true}).click();
    await expect(next.getByLabel('Seed',{exact:true})).toHaveValue('7');
    await next.getByText('Import decklist files',{exact:true}).click();
    await next.getByLabel('Use imported decks for the next game').check();
    await next.getByLabel('Your deck source',{exact:true}).fill('8 Definitely not a Magic card');
    await next.getByLabel('Opponent deck source',{exact:true}).fill('8 Mountain');
    await next.getByRole('button',{name:'Start next game',exact:true}).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(next).toBeVisible();
    await expect(next.getByLabel('Your deck source',{exact:true})).toHaveValue('8 Definitely not a Magic card');
    await expect(strip.getByRole('link',{name:'Session history'})).toHaveAttribute('href',history!);
    await expect(strip).toContainText('Game 1');
    await expect(page.locator('main')).toHaveAttribute('data-frame-revision',revision!);
    await page.reload();
    await expect(strip).toContainText('Game 1');
    await expect(page.getByTestId('action-option').first()).toBeEnabled();
    await strip.getByRole('button',{name:'Edit next game',exact:true}).click();
    await expect(next.getByLabel('Seed',{exact:true})).toHaveValue('7');
    await expect(next.getByLabel('Your deck source',{exact:true})).toHaveCount(0);
    await next.getByRole('button',{name:'Start next game',exact:true}).click();
    await expect(next).toHaveCount(0);
    await expect(strip).toContainText('Game 2');
    await expect(strip.getByRole('link',{name:'Session history'})).toHaveAttribute('href',history!);

    await page.getByRole('link',{name:'New game',exact:true}).first().click();
    await fresh.getByText('Import decklist files',{exact:true}).click();
    await fresh.getByLabel('Use imported decks for the next game').check();
    await fresh.getByLabel('Your deck source',{exact:true}).fill('8 Definitely not a Magic card');
    await fresh.getByLabel('Opponent deck source',{exact:true}).fill('8 Mountain');
    await fresh.getByRole('button',{name:'Start game',exact:true}).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(fresh).toBeVisible();
    await expect(fresh.getByLabel('Your deck source',{exact:true})).toHaveValue('8 Definitely not a Magic card');
    await page.reload();
    await expect(fresh.getByLabel('Seed',{exact:true})).toHaveValue('7');
    await expect(strip).toContainText('Game 2');
    await fresh.getByRole('button',{name:'Start game',exact:true}).click();
    await expect(strip).toContainText('Game 1');
    await expect(strip.getByRole('link',{name:'Session history'})).not.toHaveAttribute('href',history!);
    await strip.getByRole('button',{name:'Edit next game',exact:true}).click();
    await next.getByRole('combobox',{name:'Seat 1 controller',exact:true}).selectOption('bot');
    await expect(next.getByRole('button',{name:'Save next setup',exact:true})).toHaveCount(0);
    await expect(next.getByRole('button',{name:'Start next game',exact:true})).toHaveCount(0);
    await expect(next).toContainText('Your Play session and its current game stay available');
    await next.getByTestId('opponent-select').first().selectOption('random');
    const playHistory = await strip.getByRole('link',{name:'Session history'}).getAttribute('href');
    await page.screenshot({path:testInfo.outputPath(`next-watch-${theme}.png`)});
    await next.getByRole('button',{name:'Start new watched game',exact:true}).click();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    const watchURL = new URL(page.url());
    const watchHref = `/replay?trace=${watchURL.searchParams.get('trace')}`;
    await page.reload();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    await page.goto(playHistory!);
    await expect(page.getByRole('link',{name:'Return to current game',exact:true})).toHaveAttribute('href',watchHref);
    await expect(page.getByRole('region',{name:'Session results'})).toContainText('1 active');
    // Reading the old session cannot replace this tab's watched-game destination.
    await page.getByRole('link',{name:'Return to current game',exact:true}).click();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    await page.getByRole('link',{name:'New game',exact:true}).first().click();
    await expect(fresh).toBeVisible();
    await expect(page.getByRole('link',{name:'Current game',exact:true}).first()).toHaveAttribute('href',watchHref);
    await fresh.getByRole('button',{name:'Cancel',exact:true}).click();
    await expect(page.getByTestId('position-counter')).toBeVisible();
    expect(new URL(page.url()).searchParams.get('trace')).toBe(watchURL.searchParams.get('trace'));
  });
}
