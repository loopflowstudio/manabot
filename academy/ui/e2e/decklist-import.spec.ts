import { expect, test } from '@playwright/test';

// Real native admission; no mocked card registry or setup service.
test('correct a file, start a novel deck and reconnect to its pending decision', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByText('Import decklist files', { exact: true }).click();
  await page.getByLabel('Use imported decks for the next game').check();
  await page.getByLabel('Your deck file', { exact: true }).setInputFiles({
    name: 'my-deck.txt', mimeType: 'text/plain',
    buffer: Buffer.from('\ufeff# retained source\n10 Forest\n0 Mountain'),
  });
  await page.getByLabel('Opponent deck file', { exact: true }).setInputFiles({
    name: 'opponent.txt', mimeType: 'text/plain', buffer: Buffer.from('20 Island'),
  });
  await page.getByRole('button', { name: 'Check decklists', exact: true }).click();
  await expect(page.getByText(/Your deck, line 3: Quantity/)).toBeVisible();
  await expect(page.getByLabel('Your deck source', { exact: true })).toHaveValue('\ufeff# retained source\n10 Forest\n0 Mountain');
  await page.getByLabel('Your deck source', { exact: true }).fill('# retained source\n10 Forest\n10 Mountain\nSideboard\n1 Island');
  await page.getByRole('button', { name: 'Check decklists', exact: true }).click();
  await expect(page.getByText(/Ready — development-v1/)).toBeVisible();
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeVisible();
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  const choices = await page.getByTestId('action-option').allTextContents();
  await page.reload();
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', revision!);
  await expect(page.getByTestId('action-option')).toHaveText(choices);
  await page.getByTestId('action-option').first().click();
  await expect(page.locator('main')).not.toHaveAttribute('data-frame-revision', revision!);
});
