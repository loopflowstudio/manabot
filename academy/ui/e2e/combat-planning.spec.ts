import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import type { ExperienceFrame } from '../src/lib/types';

declare global { interface Window { __combatFrame: ExperienceFrame | null; __combatSent: number; __combatSocket: WebSocket; __dropCombatReply: boolean; __delayCombatValidation: boolean; __heldCombatValidation: string[]; __lastCombatValidation: string; __lastCombatCommand: string; __combatOutcome: string; __combatChildren: string[] } }

async function start(page: Page, stage: 'attackers' | 'blockers', dense = false): Promise<void> {
  await page.addInitScript(({ stage, dense }) => {
    window.__combatFrame = null;
    window.__combatSent = 0;
    window.__dropCombatReply = false;
    window.__delayCombatValidation = false;
    window.__heldCombatValidation = [];
    window.__combatOutcome = '';
    window.__combatChildren = [];
    const Native = window.WebSocket;
    class CombatSocket extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        window.__combatSocket = this;
        this.addEventListener('message', event => {
          const payload = JSON.parse(String(event.data));
          if (payload.type === 'declaration_validation') {
            window.__lastCombatValidation = String(event.data);
            if (window.__delayCombatValidation) {
              window.__heldCombatValidation.push(String(event.data));
              event.stopImmediatePropagation();
              return;
            }
          }
          const receipt = payload.update?.receipt ?? payload.receipt;
          if (receipt?.declaration) {
            window.__combatChildren = receipt.declaration.child_command_ids;
            window.__combatOutcome = payload.status;
            if (window.__dropCombatReply) {
              window.__dropCombatReply = false;
              event.stopImmediatePropagation();
              this.close();
              return;
            }
          }
          const frame = payload.recovery?.frame ?? payload.update?.frame ?? payload.frame;
          if (frame) window.__combatFrame = frame;
        });
      }
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.type === 'new_game') {
            message.config = { ...message.config, combat_fixture: stage, dense };
            data = JSON.stringify(message);
          }
          if (message.type === 'declaration') { window.__combatSent += 1; window.__lastCombatCommand = String(data); }
        }
        super.send(data);
      }
    }
    window.WebSocket = CombatSocket;
  }, { stage, dense });
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).click();
  await expect(page.getByTestId('combat-composer')).toBeVisible();
  await page.getByRole('group', { name: 'Table interaction mode' }).getByRole('button', { name: 'Play', exact: true }).click();
}

for (const [width, height, theme] of [[1280, 800, 'dark'], [1280, 800, 'light'], [1440, 900, 'dark'], [1440, 900, 'light']] as const) {
  test(`attack plan, Inspect, keyboard confirmation and reload ${width} ${theme}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height });
    await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
    await start(page, 'attackers');
    const before = await page.evaluate(() => window.__combatFrame!);
    const creature = page.getByTestId('game-board').getByRole('button', { name: /^Boggart Brute/ }).first();
    await creature.click();
    await page.getByRole('button', { name: 'Target Passive', exact: true }).click();
    await expect(page.getByTestId('combat-summary')).toContainText('1 planned');
    await expect(creature).toHaveAccessibleName(/Planned attack/);
    await expect(creature.getByText('Planned attack', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: /Select all eligible to attack/ }).click();
    await expect(page.getByTestId('combat-summary')).toContainText('3 planned');
    await expect(page.getByTestId('game-board').locator('[data-combat-plan]')).toHaveCount(3);
    const assignments = page.getByTestId('combat-composer').getByRole('combobox');
    await assignments.first().selectOption('');
    await expect(page.getByTestId('combat-summary')).toContainText('2 planned');
    await expect(page.getByTestId('game-board').locator('[data-combat-plan]')).toHaveCount(2);
    const modes = page.getByRole('group', { name: 'Table interaction mode' });
    await modes.getByRole('button', { name: 'Inspect', exact: true }).click();
    await expect(assignments.first()).toBeDisabled();
    await expect(page.getByTestId('combat-summary')).toContainText('2 planned');
    await expect(page.getByTestId('game-board').locator('[data-combat-plan]')).toHaveCount(2);
    await modes.getByRole('button', { name: 'Play', exact: true }).click();
    await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
    expect(await page.evaluate(() => window.__combatFrame!.revision)).toBe(before.revision);
    await expect(page.getByTestId('game-board').getByRole('button', { name: /^Gray Ogre/ }).first()).toBeInViewport({ ratio: 1 });
    await expect(creature).toBeInViewport({ ratio: 1 });
    await page.getByText('History and settings', { exact: true }).click();
    const context = page.getByTestId('decision-context');
    await expect(context).toContainText('then confirm once');
    await expect(context).not.toContainText('Choose each attack declaration');
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    await page.getByText('History and settings', { exact: true }).click();
    await page.screenshot({ path: info.outputPath(`combat-${width}-${theme}.png`) });
    const confirm = page.getByRole('button', { name: 'Confirm attackers', exact: true });
    await confirm.focus();
    await page.keyboard.press('Enter');
    await expect.poll(() => page.evaluate(() => window.__combatFrame?.revision)).toBeGreaterThan(before.revision);
    expect(await page.evaluate(() => window.__combatSent)).toBe(1);
    const committed = await page.evaluate(() => window.__combatFrame!.frame_hash);
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__combatFrame?.frame_hash)).toBe(committed);
    expect(await page.evaluate(() => window.__combatSent)).toBe(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  });
}

test('menace stays editable, dense choices remain reachable and reload discards only drafts', async ({ page }, info) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await start(page, 'blockers', true);
  const before = await page.evaluate(() => window.__combatFrame!);
  const assignments = page.getByTestId('combat-composer').getByRole('combobox');
  await expect(assignments).toHaveCount(24);
  const menace = await assignments.first().locator('option').evaluateAll(options => options.find(option => option.textContent?.includes('Boggart Brute'))?.getAttribute('value'));
  expect(menace).toBeTruthy();
  await assignments.first().selectOption(menace!);
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'invalid');
  await expect(page.getByTestId('combat-validation')).toContainText('Edit the assignments before confirming');
  await expect(page.getByRole('button', { name: 'Retry validation' })).toHaveCount(0);
  await expect(page.getByTestId('combat-composer').locator('details').filter({ hasText: 'Combat requirements' })).toHaveAttribute('open', '');
  await expect(page.getByText('Boggart Brute needs at least two blockers if blocked.', { exact: true }).first()).toBeVisible();
  await expect(page.getByTestId('game-board').getByRole('button', { name: /Planned block/ })).toHaveCount(1);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.screenshot({ path: info.outputPath('invalid-menace.png') });
  await expect(page.getByRole('button', { name: 'Confirm blockers' })).toBeDisabled();
  expect(await page.evaluate(() => window.__combatFrame!.revision)).toBe(before.revision);
  await assignments.nth(1).selectOption(menace!);
  await expect(page.getByTestId('game-board').getByRole('button', { name: /Planned block/ })).toHaveCount(2);
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  const next = page.getByTestId('game-board').getByRole('button', { name: 'Next creatures', exact: true });
  for (let pageIndex = 0; pageIndex < 4; pageIndex++) await next.click();
  await expect(next).toBeDisabled();
  await expect(page.locator('[data-battlefield-lane="creatures"]').last().getByRole('status')).toHaveText('5/5');
  await assignments.last().focus();
  await expect(assignments.last()).toBeInViewport();
  await page.screenshot({ path: info.outputPath('dense-blocks.png') });
  await page.reload();
  await expect(page.getByTestId('combat-composer')).toBeVisible();
  await expect(page.getByTestId('combat-summary')).toContainText('0 planned');
  await expect(page.getByTestId('game-board').locator('[data-combat-plan]')).toHaveCount(0);
  expect(await page.evaluate(() => window.__combatFrame!.revision)).toBe(before.revision);
});


test('a delayed valid response cannot approve a changed menace draft', async ({ page }) => {
  await start(page, 'blockers');
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  const old = await page.evaluate(() => { window.__delayCombatValidation = true; return window.__lastCombatValidation; });
  const choice = page.getByTestId('combat-composer').getByRole('combobox').first();
  const menace = await choice.locator('option').evaluateAll(options => options.find(option => option.textContent?.includes('Boggart Brute'))!.getAttribute('value')!);
  await choice.selectOption(menace);
  await expect.poll(() => page.evaluate(() => window.__heldCombatValidation.length)).toBe(1);
  await page.evaluate(old => {
    window.__delayCombatValidation = false;
    window.__combatSocket.dispatchEvent(new MessageEvent('message', { data: old }));
  }, old);
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'pending');
  await expect(page.getByRole('button', { name: 'Confirm blockers' })).toBeDisabled();
  await page.evaluate(() => window.__combatSocket.dispatchEvent(new MessageEvent('message', { data: window.__heldCombatValidation[0] })));
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'invalid');
});

test('a lost committed reply recovers and retry returns the same child commitment', async ({ page }) => {
  await start(page, 'attackers');
  await page.getByRole('button', { name: /Select all eligible to attack/ }).click();
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  const revision = await page.evaluate(() => { window.__dropCombatReply = true; return window.__combatFrame!.revision; });
  await page.getByRole('button', { name: 'Confirm attackers' }).click();
  await expect.poll(() => page.evaluate(() => window.__combatFrame?.revision)).toBeGreaterThan(revision);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  const children = await page.evaluate(() => window.__combatChildren);
  const committed = await page.evaluate(() => window.__combatFrame!.frame_hash);
  expect(children).toHaveLength(3);
  await page.evaluate(() => { window.__combatOutcome = ''; window.__combatSocket.send(window.__lastCombatCommand); });
  await expect.poll(() => page.evaluate(() => window.__combatOutcome)).toBe('duplicate');
  expect(await page.evaluate(() => window.__combatChildren)).toEqual(children);
  expect(await page.evaluate(() => window.__combatFrame!.frame_hash)).toBe(committed);
});


test('capacity keeps the draft without promising that an unchanged retry can validate it', async ({ page }) => {
  await start(page, 'attackers');
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  await page.evaluate(() => { window.__delayCombatValidation = true; });
  await page.getByRole('button', { name: /Select all eligible to attack/ }).click();
  await expect.poll(() => page.evaluate(() => window.__heldCombatValidation.length)).toBe(1);
  // Inject a bounded measurement rather than constructing an expensive solver state.
  await page.evaluate(() => {
    const response = JSON.parse(window.__heldCombatValidation[0]);
    response.status = 'capacity';
    response.message = 'combat_solver capacity exceeded';
    response.capacity = { resource: 'combat_solver', limit: 65536, visited: 65537 };
    window.__delayCombatValidation = false;
    window.__combatSocket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(response) }));
  });
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'capacity');
  await expect(page.getByTestId('combat-validation')).toContainText('Legality is unknown');
  await expect(page.getByTestId('combat-summary')).toContainText('3 planned');
  await expect(page.getByRole('button', { name: 'Retry validation' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Confirm attackers' })).toBeDisabled();
  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  await expect(page.getByTestId('game-board').locator('[data-combat-plan]')).toHaveCount(0);
});


test('a validation request that could not be sent offers a working retry', async ({ page }) => {
  await start(page, 'attackers');
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  // Hold only the transport readiness measurement; the authority remains live.
  await page.evaluate(() => Object.defineProperty(window.__combatSocket, 'readyState', { configurable: true, value: WebSocket.CONNECTING }));
  await page.getByRole('button', { name: /Select all eligible to attack/ }).click();
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'transport');
  await expect(page.getByRole('button', { name: 'Confirm attackers' })).toBeDisabled();
  await page.evaluate(() => Reflect.deleteProperty(window.__combatSocket, 'readyState'));
  await page.getByRole('button', { name: 'Retry validation' }).click();
  await expect(page.getByTestId('combat-validation')).toHaveAttribute('data-status', 'valid');
  await expect(page.getByTestId('combat-summary')).toContainText('3 planned');
  await expect(page.getByRole('button', { name: 'Retry validation' })).toHaveCount(0);
  expect(await page.evaluate(() => window.__combatSent)).toBe(0);
});
