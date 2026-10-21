import { expect, test, type Page } from '@playwright/test';

type Palette = {
  bg: string;
  surface: string;
  muted: string;
  field: string;
  border: string;
  text: string;
  textSecondary: string;
  accent: string;
  accentText: string;
  accentHover: string;
};

const light: Palette = {
  bg: '#ebdfc6',
  surface: '#f7f0e0',
  muted: '#ded0af',
  field: '#fcf9ee',
  border: '#c9b892',
  text: '#3a3122',
  textSecondary: '#5f553d',
  accent: '#973427',
  accentText: '#6d5a35',
  accentHover: '#ab4634',
};

const dark: Palette = {
  bg: '#191510',
  surface: '#221c14',
  muted: '#2b241a',
  field: '#2f2717',
  border: '#423a2b',
  text: '#ece4d0',
  textSecondary: '#b3a88e',
  accent: '#b24d38',
  accentText: '#c3a568',
  accentHover: '#a34531',
};

async function palette(page: Page): Promise<Palette> {
  return page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    const token = (name: string) => style.getPropertyValue(name).trim();
    return {
      bg: token('--bg'),
      surface: token('--bg-surface'),
      muted: token('--bg-muted'),
      field: token('--bg-field'),
      border: token('--border'),
      text: token('--text'),
      textSecondary: token('--text-secondary'),
      accent: token('--accent'),
      accentText: token('--accent-text'),
      accentHover: token('--accent-hover'),
    };
  });
}

test('Sepia Etude parchment palette and typography resolve in light mode', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await page.evaluate(() => document.fonts.ready);

  expect(await palette(page)).toEqual(light);
  await expect(page.locator('body')).toHaveCSS('font-family', /Lato/);
  await expect(page.getByTestId('brand-name')).toHaveCSS('font-family', /Cormorant Garamond/);

  const newGame = page.getByRole('button', { name: 'New Game' }).first();
  await expect(newGame).toHaveCSS('background-color', 'rgb(151, 52, 39)');
  expect((await newGame.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(32);
  await newGame.focus();
  await expect(newGame).toHaveCSS('outline-color', 'rgb(151, 52, 39)');
});

test('the library-after-dark palette lifts Mountain red for dark mode', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');

  expect(await palette(page)).toEqual(dark);
  // The banner is a fixed rich world in both modes; the brand is literal ivory.
  await expect(page.getByTestId('brand-name')).toHaveCSS('color', 'rgb(248, 241, 224)');
  await expect(page.getByRole('button', { name: 'New Game' }).first()).toHaveCSS(
    'background-color',
    'rgb(178, 77, 56)',
  );
});

test('card name plate stays legible on the light palette', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 15_000 });
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'New Game' }).first().click();
  await expect(page.getByTestId('game-board')).toBeVisible({ timeout: 15_000 });

  // Card art palettes are fixed dark colors, so the name plate must keep its
  // literal dark scrim and light text even when the adaptive palette is parchment.
  const treatment = page.locator('[data-testid="card-treatment"]').first();
  await expect(treatment).toBeVisible();
  const cardName = await treatment.getAttribute('data-card-name');
  expect(cardName).toBeTruthy();
  const name = treatment.getByText(cardName as string, { exact: true }).first();
  await expect(name).toHaveCSS('color', 'rgb(250, 248, 245)');
  await expect(name.locator('..')).toHaveCSS('background-image', /rgba?\(10, 13, 20/);
});

test('touch controls meet the shared 44px target', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  const newGame = page.getByRole('button', { name: 'New Game' }).first();
  expect((await newGame.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(44);
});
