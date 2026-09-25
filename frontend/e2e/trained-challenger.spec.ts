import { expect, test } from '@playwright/test';

test('configured trained opponent completes both deck assignments and retains feedback', async ({ page }) => {
  test.skip(!process.env.ETUDE_PLAY_CANDIDATE, 'Requires a freshly trained local candidate');
  test.setTimeout(600_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const attempts: string[] = [];
  page.on('websocket', (socket) => socket.on('framereceived', ({ payload }) => {
    const message = JSON.parse(String(payload));
    const id = message.table?.attempt_id;
    if (id && !attempts.includes(id)) attempts.push(id);
    if (message.type === 'error') errors.push(message.message);
  }));
  await page.goto('/');
  const selector = page.getByTestId('opponent-select');
  await expect(selector).toHaveValue('checkpoint');
  const identity = await selector.locator('option:checked').textContent();
  expect(identity).toMatch(/seed \d+ · [a-f0-9]{10}/);
  for (let assignment = 0; assignment < 2; assignment++) {
    await page.getByTestId('deck-select-hero').selectOption(assignment ? 'gw_allies' : 'ur_lessons');
    await page.getByTestId('deck-select-villain').selectOption(assignment ? 'ur_lessons' : 'gw_allies');
    const start = page.getByRole('button', { name: assignment ? 'Play Again' : 'New Game', exact: true }).first();
    const previousSeq = Number(await page.locator('main').getAttribute('data-update-seq'));
    await start.focus();
    await page.keyboard.press('Enter');
    await expect.poll(async () => Number(await page.locator('main').getAttribute('data-update-seq'))).toBeGreaterThan(previousSeq);
    await expect(page.getByTestId('game-board')).toBeVisible();
    await expect(page.getByTestId('deck-names')).toContainText(identity!);
    if (!assignment) {
      await page.reload();
      await expect(page.getByTestId('deck-names')).toContainText(identity!);
    }
    const actions = page.getByTestId('action-option');
    const over = page.getByText('Game Over', { exact: true });
    for (let decision = 0; decision < 1200 && !(await over.isVisible()); decision++) {
      await expect(actions.first().or(over)).toBeVisible();
      if (await over.isVisible()) break;
      const before = Number(await page.locator('main').getAttribute('data-update-seq'));
      const labels = await actions.allTextContents();
      let pick = labels.findIndex((label) => /^Play |^Cast /.test(label));
      if (pick < 0) pick = labels.findIndex((label) => /^Pass priority/.test(label));
      if (pick < 0) pick = 0;
      await actions.nth(pick).focus();
      await page.keyboard.press('Enter');
      await expect.poll(async () => Number(await page.locator('main').getAttribute('data-update-seq'))).toBeGreaterThan(before);
    }
    await expect(over).toBeVisible();
    await page.getByRole('textbox', { name: /How did this game feel/ }).fill('Automated keyboard integration check; not human feedback.');
    await page.getByRole('button', { name: 'Save note', exact: true }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Feedback saved.' })).toBeVisible();
  }
  expect(attempts).toHaveLength(2);
  await page.getByRole('link', { name: 'Open replay', exact: true }).click();
  await expect(page).toHaveURL(/replay\?trace=/);
  await expect(page.getByTestId('game-board')).toBeVisible();
  expect(errors).toEqual([]);
  await test.info().attach('attempts.json', { body: JSON.stringify({ identity, attempts }), contentType: 'application/json' });
});
