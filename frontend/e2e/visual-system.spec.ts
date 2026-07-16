import { expect, test, type Page } from '@playwright/test';

type Palette = {
  bg: string;
  surface: string;
  muted: string;
  border: string;
  text: string;
  textSecondary: string;
  accent: string;
  accentText: string;
  accentHover: string;
};

const light: Palette = {
  bg: '#faf8f5',
  surface: '#fffdfb',
  muted: '#f3eee7',
  border: '#e3ddd5',
  text: '#1a1a1a',
  textSecondary: '#6b6b6b',
  accent: '#722f37',
  accentText: '#722f37',
  accentHover: '#8b3d47',
};

const dark: Palette = {
  bg: '#2b3036',
  surface: '#343b44',
  muted: '#3c4550',
  border: '#46505b',
  text: '#f5f1ea',
  textSecondary: '#c8c1b8',
  accent: '#9b4a54',
  accentText: '#d9949d',
  accentHover: '#b05762',
};

async function palette(page: Page): Promise<Palette> {
  return page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    const token = (name: string) => style.getPropertyValue(name).trim();
    return {
      bg: token('--bg'),
      surface: token('--bg-surface'),
      muted: token('--bg-muted'),
      border: token('--border'),
      text: token('--text'),
      textSecondary: token('--text-secondary'),
      accent: token('--accent'),
      accentText: token('--accent-text'),
      accentHover: token('--accent-hover'),
    };
  });
}

test('shared Loopflow palette and typography resolve in light mode', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await page.evaluate(() => document.fonts.ready);

  expect(await palette(page)).toEqual(light);
  await expect(page.locator('body')).toHaveCSS('font-family', /Lato/);
  await expect(page.getByTestId('brand-name')).toHaveCSS('font-family', /Cormorant Garamond/);

  const newGame = page.getByRole('button', { name: 'New Game' }).first();
  await expect(newGame).toHaveCSS('background-color', 'rgb(114, 47, 55)');
  expect((await newGame.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(32);
  await newGame.focus();
  await expect(newGame).toHaveCSS('outline-color', 'rgb(114, 47, 55)');
});

test('kata contrast correction resolves on the shared dark palette', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');

  expect(await palette(page)).toEqual(dark);
  await expect(page.getByTestId('brand-name')).toHaveCSS('color', 'rgb(217, 148, 157)');
  await expect(page.getByRole('button', { name: 'New Game' }).first()).toHaveCSS(
    'background-color',
    'rgb(155, 74, 84)',
  );
});

test('touch controls meet the shared 44px target', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  const newGame = page.getByRole('button', { name: 'New Game' }).first();
  expect((await newGame.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
});
