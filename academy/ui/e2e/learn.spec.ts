import { expect, test } from '@playwright/test';
import prefix from '../../../managym/academy/fixtures/learn-demo-opening-v1.json' with { type: 'json' };

for (const outcome of ['LEARN_TAKE_LESSON', 'LEARN_DISCARD', 'DECLINE_CHOICE']) test(`Learn on the board: every answer a row, lessons in the tray, reconnect, ${outcome}`, async ({ page }) => {
  await page.addInitScript((config) => {
    // The recorded opening pays by hand; Auto-tap would submit its payment before the test reads it.
    localStorage.setItem('academy.auto-tap', 'off');
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
  await page.goto('/?demo=learn');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');

  await page.getByRole('button', { name: 'Start game', exact: true }).first().click();
  await expect.poll(() => page.evaluate(() => (window as any).__learn.frame?.revision)).toBe(0);
  for (const step of prefix.prefix) {
    const button = page.locator(`[data-testid="action-option"][data-offer-id="${step.offer_id}"]`);
    await expect(button).toHaveAttribute('data-action-description', step.label);
    const revision = await page.evaluate(() => (window as any).__learn.frame.revision);
    await button.click();
    await expect.poll(() => page.evaluate(() => (window as any).__learn.frame.revision)).toBeGreaterThan(revision);
  }
  await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', 'LEARN');
  const before = await page.evaluate(() => ({ frame: (window as any).__learn.frame, count: (window as any).__learn.commands.length }));
  // Every Learn answer is a rail row; the lessons outside the game sit in the tray.
  await expect(page.getByTestId('action-option')).toHaveCount(before.frame.offers.length);
  await expect(page.getByTestId('tray')).toContainText('Accumulate Wisdom');
  for (const kind of ['LEARN_TAKE_LESSON', 'LEARN_DISCARD', 'DECLINE_CHOICE']) {
    await expect(page.locator(`[data-testid="action-option"][data-action-type="${kind}"]`).first()).toBeVisible();
  }
  // Pointing commits nothing; a reconnect returns the same decision.
  await page.getByTestId('tray').getByTestId('board-card').first().hover();
  expect(await page.evaluate(() => (window as any).__learn.commands.length)).toBe(before.count);
  await page.evaluate(() => (window as any).__learn.sockets.at(-1).close());
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', { timeout: 15_000 });
  await expect(page.getByTestId('action-option')).toHaveCount(before.frame.offers.length);
  expect(await page.evaluate(() => (window as any).__learn.frame.revision)).toBe(before.frame.revision);
  expect(await page.evaluate(() => (window as any).__learn.frame.frame_hash)).toBe(before.frame.frame_hash);
  await page.locator(`[data-testid="action-option"][data-action-type="${outcome}"]`).first().click();
  await expect.poll(() => page.evaluate(() => (window as any).__learn.commands.length)).toBe(before.count + 1);
  await expect.poll(() => page.evaluate(() => (window as any).__learn.frame.revision)).toBeGreaterThan(before.frame.revision);
  const after = await page.evaluate(() => (window as any).__learn.frame.projection);
  expect(after.opponent.hand).toEqual([]);
  expect(after.opponent.sideboard ?? []).toEqual([]);
  if (outcome === 'LEARN_TAKE_LESSON') {
    expect(after.agent.sideboard).toHaveLength(2);
    expect(after.agent.hand.some((card: any) => card.name === 'Accumulate Wisdom')).toBe(true);
    await expect(page.locator('.hand [data-testid=board-card][data-card-name="Accumulate Wisdom"]')).toBeVisible();
  } else {
    expect(after.agent.sideboard).toHaveLength(3);
  }
  expect(errors).toEqual([]);
});
