import type { Page } from '@playwright/test';

export async function selectDeck(page: Page, seat: 'hero' | 'villain', deck: string): Promise<void> {
  const browser = page.getByTestId('deck-browser');
  if (!await browser.isVisible()) {
    const open = page.getByText('Browse bundled decks and cards', {exact:true});
    // A deck pool on either seat hides the bundled browser; the Elemental preset returns both seats to bundled decks.
    if (!await open.isVisible()) await page.getByRole('button', { name: 'Elemental', exact: true }).click();
    await open.click();
  }
  const row = browser.getByTestId(`browse-${deck}`);
  if (!deck.startsWith('elemental_') && !await row.isVisible()) {
    await browser.getByText('Other decks', { exact: true }).click();
  }
  await row.click();
  await browser.getByTestId(`assign-deck-${seat}`).click();
}
