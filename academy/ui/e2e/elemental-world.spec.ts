import { selectDeck } from './deck-selection';
import { expect, test } from '@playwright/test';
import manifest from '../src/lib/packs/elemental-world/v1/manifest.json' with { type: 'json' };
import type { ExperienceFrame } from '../src/lib/types';

for (const [hero, villain] of [['elemental_ug', 'elemental_bg'], ['elemental_bg', 'elemental_ug'], ['elemental_ug', 'elemental_ug']]) {
  test(`${hero} vs ${villain}: selectors preserve Lesson pools and execute Commands`, async ({ page }) => {
    test.setTimeout(90_000);
    let frame: ExperienceFrame | null = null;
    const observeFrame = (value: unknown): void => {
      if (!value || typeof value !== 'object') return;
      if ('frame_hash' in value && 'projection' in value) frame = value as ExperienceFrame;
      else for (const child of Object.values(value)) observeFrame(child);
    };
    page.on('websocket', socket => {
      socket.on('framereceived', event => observeFrame(JSON.parse(String(event.payload))));
    });
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    for (const deck of manifest.roster) {
      await expect(page.getByTestId(`browse-${deck.deck_id}`)).toBeVisible();
    }
    await page.getByTestId('opponent-select').selectOption('passive');
    await selectDeck(page, 'hero', hero);
    await selectDeck(page, 'villain', villain);
    const previousMatch = frame?.match_id;
    await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
    await expect.poll(() => frame?.match_id).not.toBe(previousMatch);
    await expect(page.getByTestId('action-option').first()).toBeVisible();
    const current = frame!;
    expect(current.asset_pack?.id).toBe(manifest.pack.id);
    const setup = manifest.roster.find(deck => deck.deck_id === hero)!;
    expect(current.projection.agent.sideboard?.map(card => card.name).sort()).toEqual(
      Object.entries(setup.sideboard).flatMap(([name, count]) => Array(count).fill(name)).sort(),
    );
    expect(current.projection.opponent.hand).toEqual([]);
    expect(current.projection.opponent.sideboard ?? []).toEqual([]);
    await page.getByTestId('action-option').first().click();
    await expect.poll(() => frame?.revision).toBeGreaterThan(current.revision);
  });
}
