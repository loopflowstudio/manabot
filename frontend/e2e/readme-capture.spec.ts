import { expect, test, type Page } from '@playwright/test';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL9WQAAAABJRU5ErkJggg==',
  'base64',
);

function collectRuntimeErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

test('README hero shows the versioned belief-to-strategy prototype', async ({ page }) => {
  const errors = collectRuntimeErrors(page);
  await page.route('https://api.scryfall.com/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: TINY_PNG }),
  );

  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });

  const surface = page.getByTestId('decision-advice');
  await expect(surface).toBeVisible({ timeout: 15_000 });
  if ((await surface.getAttribute('open')) === null) {
    await surface.locator('summary').click();
  }

  const heldInteraction = page
    .getByTestId('advice-scenario-option')
    .filter({ hasText: 'Opponent holding interaction' });
  await heldInteraction.click();
  await expect(heldInteraction.locator('input')).toBeChecked();

  await expect(page.getByTestId('advice-facts')).toContainText('Opponent hand');
  await expect(page.getByTestId('advice-facts')).toContainText('(hidden)');
  await expect(page.getByTestId('advice-advice')).toContainText('Play Mountain');
  await expect(page.getByTestId('advice-advice')).toContainText('Pass priority');
  await expect(page.getByTestId('advice-action-row')).toHaveCount(2);
  await expect(page.getByTestId('advice-delta-row')).toHaveCount(2);
  await expect(page.getByTestId('advice-deltas')).toContainText('policy');
  await expect(page.getByTestId('advice-deltas')).toContainText('value');
  await expect(page.getByTestId('advice-footer')).toContainText('flat-mc-search-v1');
  await expect(page.getByTestId('advice-footer')).toContainText('1w-8r-16s');
  await expect(page.getByTestId('advice-footer')).toContainText(
    'advisory only — submit through the ActionPanel',
  );
  await expect(surface).toHaveScreenshot('ai-assisted-testing-house-v1.png');

  expect(errors, `browser errors:\n${errors.join('\n')}`).toEqual([]);
});
