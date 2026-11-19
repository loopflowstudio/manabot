import { expect, test, type Locator, type Page } from '@playwright/test';
import type { Command, ExperienceFrame } from '../src/lib/types';

// Payment on the board: the automatic payer's plan is drawn on the lands it
// would tap, the primary button reads it, and tapping a land yourself is the
// authority's own offer. Previewing never sends a Command.

declare global {
  interface Window { __mana: { frame: ExperienceFrame | null; commands: Command[]; previews: number } }
}
async function track(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.__mana = { frame: null, commands: [], previews: 0 };
    const Native = window.WebSocket;
    class PaymentSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        this.addEventListener('message', event => {
          const payload = JSON.parse(String(event.data));
          const frame = payload.recovery?.frame ?? payload.update?.frame ?? payload.frame;
          if (frame) window.__mana.frame = frame;
        });
      }
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.type === 'new_game') {
            message.config = { ...message.config, payment_fixture: true };
            data = JSON.stringify(message);
          }
          if (message.type === 'command') window.__mana.commands.push(message.command);
        }
        super.send(data);
      }
    }
    window.WebSocket = PaymentSocket;
  });
}
const frame = (page: Page): Promise<ExperienceFrame> => page.evaluate(() => window.__mana.frame!);
const commands = (page: Page): Promise<number> => page.evaluate(() => window.__mana.commands.length);
const planned = (page: Page): Locator => page.locator('.board [data-testid=board-card][data-payment-source]');
const rail = (page: Page, text: string | RegExp): Locator => page.getByTestId('action-option').filter({ hasText: text });

async function waitForRevision(page: Page, before: number): Promise<void> {
  await expect.poll(async () => (await frame(page)).revision, { timeout: 20_000 }).toBeGreaterThan(before);
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
}
async function clickCard(card: Locator): Promise<void> {
  // A copy under another in its pile shows only its right edge.
  const box = (await card.boundingBox())!;
  await card.click({ position: { x: box.width - 6, y: box.height / 2 } });
}

for (const [width, height] of [[1280, 800], [1440, 900]]) {
  test(`pay through the plan drawn on the lands at ${width}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height });
    await track(page);
    await page.goto('/?setup=1');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'Start game', exact: true }).click();
    await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');

    // Restricted floating mana says what it is beside its pips.
    let before = (await frame(page)).revision;
    await rail(page, 'White Lotus Hideout').first().click();
    await waitForRevision(page, before);
    before = (await frame(page)).revision;
    await rail(page, /Add \{U\} .*Lesson/).first().click();
    await waitForRevision(page, before);
    const pool = page.locator('[data-plate-side=you]');
    await expect(pool.getByTestId('mana-pool')).toContainText('Floating');
    await expect(pool.getByTestId('mana-pool-class')).toContainText('Lesson');

    // Pointing at the spell shows its plan on the lands and sends nothing.
    const sent = await commands(page);
    const strike = page.locator('.hand [data-testid=board-card][data-card-name="Lightning Strike"]');
    await strike.hover();
    await expect(planned(page)).toHaveCount(2);
    await expect(planned(page).first()).toHaveAttribute('aria-label', /would tap for \{[WUBRGC]\}/);
    expect(await commands(page)).toBe(sent);

    // Cast it on the card, target the opponent on their plate, then pay.
    before = (await frame(page)).revision;
    await strike.click();
    await waitForRevision(page, before);
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'CHOOSE_TARGET');
    before = (await frame(page)).revision;
    await page.locator('[data-plate-side=opponent] [data-testid=player-target]').click();
    await waitForRevision(page, before);
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PAY_OR_NOT');
    const primary = page.getByTestId('primary-action');
    await expect(primary).toHaveText(/^Pay \{1\}\{R\}/);
    await expect(primary).toHaveAccessibleName(/\{1\}\{R\} · \S+/);
    await expect(planned(page)).toHaveCount(2);
    await expect(page.getByTestId('decision-hint')).toContainText('Space taps the dashed lands');
    await expect(page.getByTestId('action-option')).toHaveCount((await frame(page)).offers.length);
    for (const colorScheme of ['dark', 'light'] as const) {
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await expect(planned(page).first()).toBeInViewport();
      await expect(primary).toBeInViewport({ ratio: 1 });
      await page.screenshot({ path: testInfo.outputPath(`payment-${width}-${colorScheme}.png`) });
    }

    // A reload finds the same pending payment and asks for its plan again.
    const paymentHash = (await frame(page)).frame_hash;
    const plannedIds = await planned(page).evaluateAll(cards => cards.map(card => card.getAttribute('data-object-id')).sort());
    await page.reload();
    await expect.poll(async () => (await frame(page))?.frame_hash).toBe(paymentHash);
    await expect(planned(page)).toHaveCount(2);
    expect(await planned(page).evaluateAll(cards => cards.map(card => card.getAttribute('data-object-id')).sort())).toEqual(plannedIds);

    // Pointing at a land outside the plan previews "tap this, then the rest"; clicking it is the authority's tap.
    // The first land outside the plan tops its pile, so its left edge is the
    // part only it covers: further right, a copy under it shares the point
    // and the hover raises whichever the browser hits first. The land and the
    // point are read together, since a second read can name another card.
    const { ownId, x, y } = await page.evaluate(() => {
      const land = [...document.querySelectorAll<HTMLElement>('.you .row.support [data-testid=board-card].legal:not([data-payment-source])')]
        .find(card => card.dataset.cardName === 'Mountain')!;
      const box = land.getBoundingClientRect();
      return { ownId: land.dataset.objectId!, x: box.left + 6, y: box.top + box.height / 2 };
    });
    await page.mouse.move(x, y);
    await expect(page.locator(`[data-object-id="${ownId}"][data-payment-source="1"]`)).toHaveCount(1);
    await expect(primary).toHaveText(/^Pay \{1\}\{R\}\s*Space$/);
    before = (await frame(page)).revision;
    await clickCard(page.locator(`[data-object-id="${ownId}"]`));
    await waitForRevision(page, before);
    expect((await page.evaluate(() => window.__mana.commands.at(-1)!)).offer_id).toBeDefined();
    await page.mouse.move(2, 2);
    await expect(page.locator(`[data-object-id="${ownId}"]`)).toHaveAttribute('aria-label', /tapped/);

    // Space pays what remains along the plan the button reads.
    const after = await frame(page);
    if (after.action_space === 'PAY_OR_NOT') {
      // The tap paid {R}; the plan for the rest is one more source.
      await expect(primary).toHaveAccessibleName(/\{1\}\{R\} · \S+/);
      await expect(planned(page)).toHaveCount(1);
      const remaining = await planned(page).evaluateAll(cards => cards.map(card => card.getAttribute('data-object-id')));
      before = after.revision;
      await page.keyboard.press('Space');
      await waitForRevision(page, before);
      for (const id of remaining) await expect(page.locator(`[data-object-id="${id}"]`)).toHaveAttribute('aria-label', /tapped/);
    }
    await expect(page.getByTestId('stack-entry').filter({ hasText: 'Lightning Strike' })).toHaveCount(1);
    await expect(planned(page)).toHaveCount(0);
  });
}
