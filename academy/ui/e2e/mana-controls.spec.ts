import { expect, test } from '@playwright/test';
import { finishBoardNarration } from './presentation-readiness';

for (const [width, height] of [[1280, 800], [1440, 900]]) {
  for (const colorScheme of ['dark', 'light'] as const) {
    test(`float and undo at ${width} × ${height} in ${colorScheme}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await page.addInitScript(() => {
        const send = WebSocket.prototype.send;
        WebSocket.prototype.send = function (data) {
          if (typeof data === 'string') {
            const message = JSON.parse(data);
            if (message.type === 'new_game') {
              message.config.mana_controls_fixture = true;
              data = JSON.stringify(message);
            }
          }
          send.call(this, data);
        };
      });
      await page.goto('/');
      await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 20_000 });
      await page.getByRole('button', { name: 'Start game', exact: true }).press('Enter');
      const mana = page.getByRole('region', { name: 'Mana', exact: true });
      await expect(mana).toBeVisible();
      await finishBoardNarration(page);

      const cast = page.getByRole('button', { name: 'Cast Gray Ogre', exact: true });
      const sources = page.getByRole('list', { name: 'Would tap' }).getByRole('listitem');
      await cast.focus();
      await expect(sources).toHaveCount(3);
      const original = await sources.allTextContents();
      const forest = mana.getByRole('button', { name: /Forest.*Tap for mana/ });
      if (colorScheme === 'light') {
        await page.getByRole('button', { name: 'Play', exact: true }).click();
        await page.locator('[data-object-id][aria-label^="Forest"]').click();
        await page.getByRole('button', { name: 'Tap for mana: {G}', exact: true }).click();
      } else {
        await forest.press('Enter');
      }
      const undo = mana.getByRole('button', { name: 'Undo: Forest tapped for {G}' });
      await expect(undo).toBeFocused();
      const pool = page.getByTestId('mana-pool').last().locator('summary');
      await expect(pool).toContainText('1 {G}');
      await expect(page.locator('[data-object-id][aria-label^="Forest"]')).toHaveAttribute('aria-label', /tapped/);
      await cast.focus();
      await expect(sources).toHaveCount(2);
      if (colorScheme === 'light') await undo.click();
      else await undo.press('Enter');
      await expect(forest).toBeFocused();
      await expect(pool).toContainText('Empty');
      await cast.focus();
      await expect(sources).toHaveText(original);

      await page.getByTestId('action-option').filter({ hasText: 'Treasure' }).press('Enter');
      const white = page.getByRole('button', { name: 'Add {W}', exact: true });
      await expect(white).toBeFocused();
      await page.keyboard.press('Enter');
      const refused = mana.getByRole('button', { name: /^Undo:/ });
      await expect(refused).toHaveAttribute('aria-disabled', 'true');
      await expect(refused).toHaveAccessibleDescription('Treasure was sacrificed for this mana.');
      await expect(refused).toBeFocused();
      const revision = await page.getByTestId('decision-surface').getAttribute('data-revision');
      await page.keyboard.press('Enter');
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', revision!);
    });
  }
}
