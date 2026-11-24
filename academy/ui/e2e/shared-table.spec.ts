import {expect, test, type Page} from '@playwright/test';

/** Sharing and Study tools live in the top bar's Menu. */
async function openTools(page: Page): Promise<void> {
  const menu = page.locator('details.game-menu');
  if (await menu.getAttribute('open') === null) await page.getByTestId('game-menu').click();
  const tools = menu.locator('details').filter({has: page.getByText('Sharing and Study tools', {exact:true})});
  if (await tools.getAttribute('open') === null) await page.getByText('Sharing and Study tools', {exact:true}).click();
}
async function closeMenu(page: Page): Promise<void> {
  if (await page.locator('details.game-menu').getAttribute('open') !== null) await page.getByTestId('game-menu').click();
}

test('participant access, transfer and real Study survive the simpler table', async ({browser}, testInfo) => {
  const pilotContext = await browser.newContext({permissions:['clipboard-read','clipboard-write'], reducedMotion:'reduce'});
  const watcherContext = await browser.newContext({reducedMotion:'reduce'});
  try {
    const pilot = await pilotContext.newPage();
    await pilot.goto('/?setup=1');
    await expect(pilot.getByTestId('connection-badge')).toHaveText('connected');
    await pilot.getByTestId('opponent-select').selectOption('passive');
    await pilot.getByRole('button', {name:'Start game', exact:true}).click();
    await openTools(pilot);
    await pilot.getByTestId('copy-watcher-invite').click();
    await closeMenu(pilot);
    const invite = await pilot.evaluate(() => navigator.clipboard.readText());
    const watcher = await watcherContext.newPage();
    await watcher.goto(invite);
    await openTools(watcher);
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the watcher');
    await closeMenu(watcher);
    await expect(watcher.getByTestId('action-option').first()).toBeDisabled();
    await expect(pilot.getByTestId('author-belief')).toHaveCount(0);
    await expect(pilot.getByTestId('decision-advice')).toHaveCount(0);
    const before = await pilot.locator('main').getAttribute('data-frame-revision');
    await pilot.getByTestId('action-option').filter({hasText:'Keep hand'}).click();
    await expect(watcher.locator('main')).not.toHaveAttribute('data-frame-revision',before!);
    await openTools(watcher);
    const decision = watcher.getByTestId('restore-table-decision').first();
    await expect(decision).toBeVisible();
    await decision.click();
    await expect(watcher.getByTestId('return-to-live-table')).toBeVisible();
    await closeMenu(watcher);
    await watcher.getByTestId('action-option').filter({hasText:'Keep hand'}).first().click();
    await openTools(watcher);
    await expect(watcher.getByTestId('return-from-participant-branch')).toBeVisible();
    await watcher.getByTestId('return-from-participant-branch').click();
    await watcher.getByTestId('return-to-live-table').click();
    await openTools(pilot);
    await pilot.getByTestId('transfer-pilot').click();
    await closeMenu(pilot);
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the pilot');
    await closeMenu(watcher);
    await expect(pilot.getByTestId('action-option').first()).toBeDisabled();
    await expect(watcher.getByTestId('action-option').first()).toBeEnabled();
    await watcher.reload();
    await openTools(watcher);
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the pilot');
    await watcher.screenshot({path:testInfo.outputPath('shared-table.png')});
  } finally {
    await pilotContext.close(); await watcherContext.close();
  }
});
