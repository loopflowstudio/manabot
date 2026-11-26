import { expect, test } from '@playwright/test';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

// The design system's values are checked by `src/lib/design.test.ts`; this
// checks what only a browser can: the bundled faces load and are used.
test('the interface and the game faces load from the bundle', async ({ page }) => {
  await page.goto(SETUP);
  await page.evaluate(() => document.fonts.ready);

  await expect(page.locator('body')).toHaveCSS('font-family', /^"DM Sans"/);
  await expect(page.getByTestId('brand-name')).toHaveCSS('font-family', /^Spectral/);
  const loaded = await page.evaluate(async () => ({
    ui: await document.fonts.load('500 14px "DM Sans"').then(faces => faces.length),
    game: await document.fonts.load('600 16px Spectral').then(faces => faces.length),
  }));
  expect(loaded.ui).toBeGreaterThan(0);
  expect(loaded.game).toBeGreaterThan(0);
});

test('touch controls meet the shared 44px target', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(SETUP);

  const newGame = page.getByRole('button', { name: 'Start game' }).first();
  expect((await newGame.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
});
