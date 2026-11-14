import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const proof = new URL('../test-results/launch/addresses.json', import.meta.url);

test('starting and reopening an unfinished bot game keeps the Watch two bots heading', async ({ page }) => {
  const start = page.getByRole('button', { name: 'Start watching' });
  const heading = page.getByRole('heading', { name: 'Watch two bots', exact: true });
  const decision = page.getByText(/^Decision 1 of 1 ·/);
  await page.goto('/replay?watch=1');
  await expect(start).toBeEnabled();
  await page.getByRole('combobox', { name: 'Seat 1', exact: true }).selectOption('random');
  await page.getByRole('combobox', { name: 'Seat 2', exact: true }).selectOption('random');
  await start.click();
  await expect(decision).not.toContainText('Finished');
  await expect(heading).toBeVisible();
  await expect(start).toBeHidden();
  await page.reload();
  await expect(decision).toBeVisible();
  await expect(heading).toBeVisible();
  await expect(page.getByRole('banner').getByRole('link', { name: 'Replay', exact: true })).toHaveAttribute('aria-current', 'page');
  await expect(start).toBeHidden();
  await page.getByText('Watch another matchup', { exact: true }).click();
  await expect(start).toBeVisible();
});

test('launch replaces unrelated resume, reloads pending game and opens selected replay', async ({ page, context }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as { game: string; replay: string };
  const frames: { match_id: string; projection: unknown; prompt: unknown }[] = [];
  const errors: string[] = [];
  page.on('pageerror', error => { errors.push(error.message); console.error(error.message); });
  page.on('websocket', socket => socket.on('framereceived', event => {
    const message = JSON.parse(String(event.payload));
    const frame = message.recovery?.frame ?? message.frame;
    if (frame) frames.push(frame);
  }));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').selectOption('passive');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  const unrelated = frames.at(-1)!.match_id;
  await page.goto(addresses.game);
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  const launched = frames.at(-1)!;
  expect(launched.match_id).not.toBe(unrelated);
  expect(launched.prompt).toBeTruthy();
  await page.reload();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  expect(frames.at(-1)!.match_id).toBe(launched.match_id);
  expect(frames.at(-1)!.projection).toEqual(launched.projection);
  // Another tab has its own resume credentials and launches its own game.
  const second = await context.newPage();
  await second.goto(addresses.game);
  await expect(second.getByTestId('action-option').first()).toBeEnabled();
  await second.close();
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('opponent-select')).toHaveValue('passive');
  await expect(page.getByTestId('selected-deck-hero')).toHaveAttribute('data-deck-id', 'gw_allies_jeong');
  await expect(page.getByText('Choose your next matchup.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  expect(page.url()).not.toContain('launch=');
  await expect.poll(() => frames.at(-1)?.match_id).not.toBe(launched.match_id);
  const ordinary = frames.at(-1)!.match_id;
  await page.reload();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  expect(frames.at(-1)!.match_id).toBe(ordinary);
  await page.goto(addresses.replay);
  await expect(page.getByRole('heading', { name: 'Replay', exact: true })).toBeVisible();
  await expect(page.getByText('Seat 1: first-legal-a · Seat 2: first-legal-b', { exact: true })).toBeVisible();
  await expect(page.getByRole('banner').getByRole('link', { name: 'Replay', exact: true })).toHaveAttribute('aria-current', 'page');
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeHidden();
  await expect(page.getByTestId('deck-browser')).toBeHidden();
  await expect(page.getByTestId('game-board')).toBeVisible();
  // This tiny retained game ends at its first decision.
  await expect(page.getByText(/^Decision 1 of 1 ·/)).toBeVisible();
  const step = page.getByRole('button', { name: 'Step decision', exact: true });
  await expect(step).toBeInViewport();
  await expect(step).toBeDisabled();
  await page.screenshot({ path: 'test-results/launch/replay-landing.png' });
  await page.getByText('Watch another matchup', { exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start watching' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('imported launch setup shows selected sources and switches back to bundled decks', async ({ page }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as { imported_game: string };
  await page.goto(addresses.imported_game);
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  const imports = page.getByRole('checkbox', { name: 'Use imported decks for the next game' });
  await expect(imports).toBeVisible();
  await expect(imports).toBeChecked();
  await expect(page.getByTestId('deck-browser')).toHaveCount(0);
  await expect(page.getByLabel('Your deck source', { exact: true })).toHaveValue(/Mountain/);
  await expect(page.getByLabel('Opponent deck source', { exact: true })).toHaveValue('20 Mountain\n');
  await expect(page.getByText('Selected source: my-deck.txt', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/launch/imported-setup.png', fullPage: true });
  await imports.uncheck();
  await expect(page.getByTestId('deck-browser')).toBeVisible();
  await imports.check();
  await expect(page.getByTestId('deck-browser')).toHaveCount(0);
  await expect(page.getByLabel('Opponent deck source', { exact: true })).toHaveValue('20 Mountain\n');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-option').first()).toBeEnabled();
  expect(page.url()).not.toContain('launch=');
  await expect(page.getByTestId('deck-names')).toContainText('Imported deck');
});


test('an exported collector game opens with both policy names, training label and default art', async ({ page }) => {
  const addresses = JSON.parse(readFileSync(proof, 'utf8')) as { training_replay: string };
  await page.goto(addresses.training_replay);
  await expect(page.getByRole('heading', { name: 'Replay', exact: true })).toBeVisible();
  await expect(page.getByText('Training game', { exact: true })).toBeVisible();
  await expect(page.getByText('Seat 1: learner · Seat 2: learner', { exact: true })).toBeVisible();
  const visibleCard = page.locator('[data-testid="card-treatment"]:visible').first();
  await visibleCard.scrollIntoViewIfNeeded();
  await expect(visibleCard).toHaveAttribute('data-art-source', 'scryfall');
  await expect(page.getByText('Card art credits', { exact: true })).toBeVisible();
  await page.getByText('Card art credits', { exact: true }).click();
  await expect(page.getByRole('contentinfo')).toContainText('Proof Artist');
  // Failed art remains usable and has an identifiable procedural fallback.
  await page.route('**/api/card-art/**', route => route.fulfill({ status: 404 }));
  await page.reload();
  await expect(page.locator('[data-testid="card-treatment"]:visible').first()).toHaveAttribute('data-art-source', 'procedural');
});
