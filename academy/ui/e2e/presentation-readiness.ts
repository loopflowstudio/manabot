import { expect, type Page } from '@playwright/test';

// CSS animation suppression does not stop the narration's JavaScript timers.
// Finish through the existing control, even behind the terminal modal, without
// moving focus or discarding the retained consequence history.
export async function finishBoardNarration(page: Page): Promise<void> {
  const presentationStage = page.getByTestId('presentation-stage');
  const finished = await page.evaluate(() => {
    const stage = document.querySelector<HTMLElement>('[data-testid="presentation-stage"]');
    if (!stage) {
      return false;
    }
    const finish = [...stage.querySelectorAll('button')].find(
      (button) => button.textContent?.trim() === 'Finish',
    );
    if (!(finish instanceof HTMLButtonElement)) {
      throw new Error('presentation stage has no Finish control');
    }
    finish.click();
    return true;
  });
  if (finished) {
    await expect(presentationStage).toBeHidden();
  }
}
