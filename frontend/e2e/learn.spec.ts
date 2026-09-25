import { expect, test } from '@playwright/test';
import prefix from '../../etude/fixtures/learn-demo.json' with { type: 'json' };

test('Learn modes, previews, Back, reconnect and three ordinary outcomes', async ({ page }) => {
  await page.addInitScript((config) => {
    const state = { commands: [] as unknown[], sockets: [] as WebSocket[], frame: null as any };
    const Native = window.WebSocket;
    class TrackedSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        state.sockets.push(this);
        this.addEventListener('message', event => {
          const findFrame = (value: any): void => {
            if (!value || typeof value !== 'object') return;
            if (value.frame_hash && value.projection) state.frame = value;
            else for (const child of Object.values(value)) findFrame(child);
          };
          findFrame(JSON.parse(event.data));
        });
      }
      send(data: any): void {
        const message = JSON.parse(data);
        if (message.type === 'new_game') message.config = config;
        if (message.type === 'command') state.commands.push(message.command);
        super.send(JSON.stringify(message));
      }
    }
    (window as any).__learn = state;
    window.WebSocket = TrackedSocket;
  }, prefix.config);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');

  for (const outcome of ['LEARN_TAKE_LESSON', 'LEARN_DISCARD', 'DECLINE_CHOICE']) {
    await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
    for (const step of prefix.prefix) {
      const button = page.locator(`[data-testid="action-option"][data-offer-id="${step.offer_id}"]`);
      await expect(button).toHaveAttribute('aria-label', step.label);
      const revision = await page.evaluate(() => (window as any).__learn.frame.revision);
      await button.click();
      await expect.poll(() => page.evaluate(() => (window as any).__learn.frame.revision)).toBeGreaterThan(revision);
    }
    await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', 'LEARN');
    const before = await page.evaluate(() => ({ frame: (window as any).__learn.frame, count: (window as any).__learn.commands.length }));
    await page.getByRole('button', { name: 'Take a Lesson', exact: true }).click();
    await expect(page.getByTestId('learn-card-preview')).toHaveCount(3);
    await expect(page.getByTestId('learn-card-preview').first().getByTestId('card-treatment')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Take Accumulate Wisdom', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Back', exact: true }).click();
    await page.getByRole('button', { name: 'Discard and draw', exact: true }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('learn-card-preview').first()).toBeVisible();
    await expect(page.getByTestId('learn-card-preview').first().getByTestId('card-treatment')).toBeVisible();
    expect(await page.evaluate(() => (window as any).__learn.commands.length)).toBe(before.count);
    expect(await page.evaluate(() => (window as any).__learn.frame.frame_hash)).toBe(before.frame.frame_hash);
    await page.evaluate(() => (window as any).__learn.sockets.at(-1).close());
    await expect(page.getByRole('button', { name: 'Take a Lesson', exact: true })).toBeVisible({ timeout: 15_000 });
    expect(await page.evaluate(() => (window as any).__learn.frame.revision)).toBe(before.frame.revision);
    if (outcome !== 'DECLINE_CHOICE') {
      await page.getByRole('button', { name: outcome === 'LEARN_TAKE_LESSON' ? 'Take a Lesson' : 'Discard and draw', exact: true }).click();
    }
    await page.locator(`[data-testid="action-option"][data-action-type="${outcome}"]`).first().click();
    await expect.poll(() => page.evaluate(() => (window as any).__learn.commands.length)).toBe(before.count + 1);
    await expect.poll(() => page.evaluate(() => (window as any).__learn.frame.revision)).toBeGreaterThan(before.frame.revision);
    const after = await page.evaluate(() => (window as any).__learn.frame.projection);
    expect(after.opponent.hand).toEqual([]);
    expect(after.opponent.sideboard ?? []).toEqual([]);
    if (outcome === 'LEARN_TAKE_LESSON') {
      expect(after.agent.sideboard).toHaveLength(2);
      expect(after.agent.hand.some((card: any) => card.name === 'Accumulate Wisdom')).toBe(true);
      await expect(page.getByTestId('known-hand')).toContainText('Accumulate Wisdom');
    } else {
      expect(after.agent.sideboard).toHaveLength(3);
    }
  }
  expect(errors).toEqual([]);
});

test('checked Learn demo opens directly into normal browser controls', async ({ page }) => {
  await page.goto('/?demo=learn');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByRole('button', { name: 'Take a Lesson', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Take a Lesson', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Take Accumulate Wisdom', exact: true })).toBeVisible();
});
