import type { Page } from '@playwright/test';

export async function selectDeck(page: Page, seat: 'hero' | 'villain', deck: string): Promise<void> {
  const browser = page.getByTestId('deck-browser');
  const row = browser.getByTestId(`browse-${deck}`);
  if (!deck.startsWith('elemental_') && !await row.isVisible()) {
    await browser.getByText('Other decks', { exact: true }).click();
  }
  await row.click();
  await browser.getByTestId(`assign-deck-${seat}`).click();
}
