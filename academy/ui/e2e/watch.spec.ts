import { expect, test } from '@playwright/test';

test('watch reads choices and consequences on the board, then returns from Study', async ({ page }) => {
  await page.goto('/?watch=1');
  await page.getByRole('button', { name: 'Start watching' }).click();
  await expect(page.getByRole('button', {name: 'Continue', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Continue', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  // The recorded choice: every offer is a row, the played one marked, evidence beside the prompt.
  const rows = page.getByTestId('action-option');
  await expect(rows.first()).toBeDisabled();
  await expect(page.locator('[data-testid="action-option"][data-chosen="true"]')).toHaveCount(1);
  await expect(page.getByTestId('recorded-decision-evidence')).toBeVisible();
  // A Watch opens with Seat 1 at the bottom and both hands up, and stays so through each step.
  const perspective = page.getByRole('combobox', {name: 'Perspective'});
  const bothHands = page.getByRole('checkbox', {name: 'Show both hands'});
  const topSeat = (): Promise<string | null> => page.locator('[data-plate-side="opponent"]').getAttribute('data-player-index');
  await expect(perspective).toHaveValue('0');
  await expect(bothHands).toBeChecked();
  await expect(page.getByTestId('replay-perspective')).toContainText('Both hands · spectator view');
  // Both hands are ribbons on the board's edges, the top one face-up: no backs, no popup on the plate.
  const topHand = page.locator('[data-testid="hand-ribbon"][data-ribbon="top"]');
  const bothFaceUp = async (): Promise<void> => {
    await expect(page.getByTestId('hand-ribbon')).toHaveCount(2);
    await expect(topHand.getByTestId('board-card')).not.toHaveCount(0);
    await expect(page.getByTestId('card-back')).toHaveCount(0);
  };
  await bothFaceUp();
  // The seat choosing is marked on its plate, and its hand is the one out.
  await expect(page.getByTestId('plate-choosing')).toHaveCount(1);
  await expect(page.locator('[data-testid="hand-ribbon"][data-out="true"]')).toHaveCount(1);
  const rowsTop = await page.locator('.half.opponent .rows').boundingBox();
  const top = await topSeat();
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
  await expect(page.getByTestId('replay-perspective')).toHaveText('Both hands · spectator view');
  await bothFaceUp();
  await page.getByRole('button', {name: 'Next', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 3 / 3');
  await expect(bothHands).toBeChecked();
  await bothFaceUp();
  expect(await topSeat()).toBe(top);
  // Stepping moved hands, never the battlefield under them.
  expect(await page.locator('.half.opponent .rows').boundingBox()).toEqual(rowsTop);
  // Show hand on the waiting plate brings its hand out beside the active one.
  await page.getByTestId('hand-toggle').filter({ hasText: 'Show hand' }).click();
  await expect(page.locator('[data-testid="hand-ribbon"][data-out="true"]')).toHaveCount(2);
  await page.getByTestId('hand-toggle').filter({ hasText: 'Tuck hand' }).first().click();
  await expect(page.locator('[data-testid="hand-ribbon"][data-out="true"]')).toHaveCount(1);
  await page.getByRole('button', {name: 'Previous', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
  await expect(rows).toHaveCount(0);
  await page.getByRole('button', {name: 'Previous', exact: true}).click();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  await page.getByRole('button', {name: 'Try another line', exact: true}).click();
  await page.getByTestId('action-option').first().click();
  await expect(page.getByText('Your line is committed')).toBeVisible();
  await page.getByTestId('study-return').click();
  await expect(page.getByRole('button', {name: 'Try another line', exact: true})).toBeFocused();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 1 / 3');
  // Study drew the acting view; the view shown before it is back.
  await expect(bothHands).toBeChecked();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
  await page.reload();
  await expect(page.getByTestId('position-counter')).toHaveText('Position 2 / 3');
});

test('opens formatted arena JSON as one game', async ({ page }) => {
  const game = { match_id: 'recorded-match', decisions: [] };
  let imported: unknown;
  await page.route('**/api/watch/import', async route => {
    imported = route.request().postDataJSON();
    await route.fulfill({ status: 400, json: { detail: 'arena_import_invalid' } });
  });
  await page.goto('/games');
  await page.getByText('Open a recorded game file',{exact:true}).click();
  await expect(page.getByLabel('Open a game file')).toHaveAccessibleDescription('An exported training game or an evaluation trace');
  await page.getByLabel('Open a game file').setInputFiles({
    name: 'game.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(game, null, 2)),
  });
  await expect(page.getByRole('alert')).toHaveText('arena import invalid');
  expect(imported).toEqual(game);
});

test('loads replay history when navigating from Watch to Replay', async ({ page }) => {
  await page.route('**/api/traces', route => route.fulfill({ json: [] }));
  await page.goto('/?watch=1');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeEnabled();
  const history = page.waitForResponse(response => new URL(response.url()).pathname === '/api/traces');
  await page.getByRole('banner').getByRole('link', { name: 'History', exact: true }).click();
  expect((await history).ok()).toBe(true);
  await expect(page.getByRole('heading', { name: 'History', exact: true })).toBeAttached();
});

test('one seat-oriented setup starts both registered bots on imported decks', async ({page}, testInfo) => {
  await page.setViewportSize({width:1280,height:800});
  await page.goto('/?watch=1');
  await expect(page.getByRole('button', {name:'Watch Bot vs Bot',exact:true})).toHaveAttribute('aria-pressed','true');
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
  // Seat 1 is always yours in Play: the same sources read as your deck and the opponent's.
  await page.getByRole('button',{name:'Play vs Bot',exact:true}).click();
  await expect(page.getByLabel('Your deck source',{exact:true})).toHaveValue('8 Island');
  await expect(page.getByLabel('Opponent deck source',{exact:true})).toHaveValue('8 Mountain');
  await page.getByRole('button',{name:'Watch Bot vs Bot',exact:true}).click();
  await expect(page.getByTestId('checkpoint-select').nth(0)).toHaveValue(`policy:${smoke[0].sha256}`);
  await expect(page.getByTestId('checkpoint-select').nth(1)).toHaveValue(`policy:${smoke[1].sha256}`);
  await page.screenshot({path:testInfo.outputPath('shared-watch-setup.png')});
  await page.getByRole('button',{name:'Start watching',exact:true}).click();
  await expect(page.getByTestId('position-counter')).toBeVisible();
  await page.getByRole('button',{name:'Continue',exact:true}).click();
  await expect(page.getByTestId('recorded-decision-evidence')).toBeVisible();
  await page.reload();
  await expect(page.getByTestId('position-counter')).toBeVisible();
});
