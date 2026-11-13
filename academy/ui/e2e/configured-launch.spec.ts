import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const proof = new URL('../test-results/launch/addresses.json', import.meta.url);

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
  await expect(page.getByTestId('game-board')).toBeVisible();
  await expect(page.getByText(/^Decision 1 of \d+/)).toBeVisible();
  expect(errors).toEqual([]);
});
