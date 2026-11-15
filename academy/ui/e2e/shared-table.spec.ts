import {expect, test} from '@playwright/test';

test('participant access, transfer and real Study survive the simpler table', async ({browser}, testInfo) => {
  const pilotContext = await browser.newContext({permissions:['clipboard-read','clipboard-write'], reducedMotion:'reduce'});
  const watcherContext = await browser.newContext({reducedMotion:'reduce'});
  try {
    const pilot = await pilotContext.newPage();
    await pilot.goto('/');
    await expect(pilot.getByTestId('connection-badge')).toHaveText('connected');
    await pilot.getByTestId('opponent-select').selectOption('passive');
    await pilot.getByRole('button', {name:'New Game', exact:true}).click();
    await pilot.getByText('Table and Study tools', {exact:true}).click();
    await pilot.getByTestId('copy-watcher-invite').click();
    const invite = await pilot.evaluate(() => navigator.clipboard.readText());
    const watcher = await watcherContext.newPage();
    await watcher.goto(invite);
    await watcher.getByText('Table and Study tools', {exact:true}).click();
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the watcher');
    await expect(watcher.getByTestId('action-option').first()).toBeDisabled();
    await expect(pilot.getByTestId('author-belief')).toHaveCount(0);
    await expect(pilot.getByTestId('decision-advice')).toHaveCount(0);
    const before = await pilot.locator('main').getAttribute('data-frame-revision');
    await pilot.getByTestId('action-option').first().click();
    await expect(watcher.locator('main')).not.toHaveAttribute('data-frame-revision',before!);
    const decision = watcher.getByTestId('restore-table-decision').first();
    await expect(decision).toBeVisible();
    await decision.click();
    await expect(watcher.getByTestId('return-to-live-table')).toBeVisible();
    await watcher.getByTestId('action-option').filter({hasText:'Pass priority'}).first().click();
    await expect(watcher.getByTestId('return-from-participant-branch')).toBeVisible();
    await watcher.getByTestId('return-from-participant-branch').click();
    await watcher.getByTestId('return-to-live-table').click();
    await pilot.getByTestId('transfer-pilot').click();
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the pilot');
    await expect(pilot.getByTestId('action-option').first()).toBeDisabled();
    await expect(watcher.getByTestId('action-option').first()).toBeEnabled();
    await watcher.reload();
    await watcher.getByText('Table and Study tools', {exact:true}).click();
    await expect(watcher.getByTestId('shared-table-panel')).toContainText('You are the pilot');
    await watcher.screenshot({path:testInfo.outputPath('shared-table.png')});
  } finally {
    await pilotContext.close(); await watcherContext.close();
  }
});
