import { expect, type Page } from '@playwright/test';

/** Confirm the composer's current combat plan (an empty plan declines) once rules validation accepts it. */
export async function confirmCombat(page: Page): Promise<void> {
  const composer = page.getByTestId('combat-composer');
  const play = page.getByRole('group', { name: 'Table interaction mode' }).getByRole('button', { name: 'Play', exact: true });
  if (await play.getAttribute('aria-pressed') === 'false') await play.click();
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  const revision = await page.locator('main').getAttribute('data-frame-revision');
  await composer.getByRole('button', { name: /^Confirm (attackers|blockers)$/ }).click();
  await expect.poll(() => page.locator('main').getAttribute('data-frame-revision')).not.toBe(revision);
}
