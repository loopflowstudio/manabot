// Optional dense-table captures; the release laptop spec asserts access and geometry.
import { expect, test } from '@playwright/test';
import { installDenseAuthority } from './fixtures/laptop-table';

test.skip(!process.env.DENSE_BOARD, 'captures run only with DENSE_BOARD=1');
for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`dense table ${viewport.width} ${colorScheme}`, async ({ browser }, testInfo) => {
      const context = await browser.newContext({ viewport, colorScheme, deviceScaleFactor: 1 });
      const page = await context.newPage();
      await installDenseAuthority(page);
      await page.goto('/');
      await expect(page.getByTestId('connection-badge')).toHaveText('connected');
      await page.getByRole('button', { name: 'New Game', exact: true }).click();
      await expect(page.getByTestId('play-workspace')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('dense-table.png') });
      await context.close();
    });
  }
}
