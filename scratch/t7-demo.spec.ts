import { expect, test } from '../academy/ui/node_modules/@playwright/test';
import AxeBuilder from '../academy/ui/node_modules/@axe-core/playwright';
import { readFileSync } from 'node:fs';
import type { ExperienceFrame } from '../academy/ui/src/lib/types';
import boltFixture from '../protocol/fixtures/bolt-target.json';

declare global {
  interface Window {
    __t7: { frame: ExperienceFrame | null; socket: WebSocket | null; statuses: string[] };
  }
}

const captured = JSON.parse(readFileSync('../../.cache/mtg226-demo/search-frames.json', 'utf8')) as {
  caster: number; paid: boolean; frames: Record<string, ExperienceFrame>;
}[];

for (const { caster, paid, frames } of captured) {
  test(`native search presentation seat ${caster + 1}, ${paid ? 'turn' : 'combat'}`, async ({ page }, info) => {
    let sendFrame: ((frame: ExperienceFrame) => void) | undefined;
    const commands: unknown[] = [];
    await page.routeWebSocket('**/ws/play', socket => {
      sendFrame = frame => socket.send(JSON.stringify({
        type: 'observation', data: frame.projection, actions: [],
        session_id: 'presentation-only', resume_token: 'presentation-only',
        recovery: { ...boltFixture.recovery, frame, presentation_tail: [], accepted_commands: [], replay_cursor: frame.revision },
      }));
      socket.onMessage(raw => {
        const message = JSON.parse(String(raw));
        if (message.type === 'new_game') sendFrame!(frames.delegated);
        if (message.type === 'command') commands.push(message.command);
      });
    });
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
    for (const stage of ['delegated', 'own_response', 'search', 'after_search']) {
      const frame = frames[stage];
      sendFrame!(frame);
      await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(frame.revision));
      for (const width of [390, 1500]) {
        await page.setViewportSize({ width, height: 1100 });
        const context = page.getByTestId('decision-context');
        if (stage === 'own_response') await expect(context).not.toContainText('chooses for');
        if (stage === 'search') {
          await expect(context).toContainText('Search cards');
          await expect(context).not.toContainText('Search your library');
          await expect(context).toContainText(`Seat ${caster + 1} chooses for Seat ${2 - caster}`);
          await expect(page.getByRole('region', { name: 'Private inspection', exact: true })).toContainText("opponent's library");
          await page.screenshot({ path: info.outputPath(`search-${width}px.png`), fullPage: true });
        }
        if (stage === 'after_search') {
          await expect(page.getByRole('region', { name: 'Private inspection', exact: true })).not.toContainText("opponent's library");
        }
        const ownMana = page.getByTestId('mana-pool').filter({ hasText: /^You mana:/ });
        const otherMana = page.getByTestId('mana-pool').filter({ hasNotText: /^You mana:/ });
        await expect(ownMana).toBeVisible();
        await expect(otherMana).toBeVisible();
        expect((await ownMana.boundingBox())!.y).toBeGreaterThan((await otherMana.boundingBox())!.y);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      }
    }
    expect(commands).toEqual([]);
  });
}

for (const width of [390, 1500]) {
  test(`live delegated cleanup and recovery at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1100 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      const state: Window['__t7'] = { frame: null, socket: null, statuses: [] };
      const Native = window.WebSocket;
      const find = (value: unknown): void => {
        if (!value || typeof value !== 'object') return;
        if ('frame_hash' in value && 'projection' in value) state.frame = value as ExperienceFrame;
        else Object.values(value).forEach(find);
      };
      class TrackedSocket extends Native {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          state.socket = this;
          this.addEventListener('message', event => {
            const message = JSON.parse(String(event.data));
            find(message);
            if (message.status) state.statuses.push(message.status);
          });
        }
        send(data: string): void {
          const message = JSON.parse(data);
          if (message.type === 'new_game') message.config = {
            villain_type: 'passive', seed: 3, auto_pass: true,
            deck_import: {
              hero: { source: '56 Island\n4 Secret of Bloodbending' },
              villain: { source: '56 Island\n4 Accumulate Wisdom' },
            },
          };
          super.send(JSON.stringify(message));
        }
      }
      window.__t7 = state;
      window.WebSocket = TrackedSocket;
    });
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
    await expect.poll(() => page.evaluate(() => Boolean(window.__t7.frame))).toBe(true);
    let frame = (await page.evaluate(() => window.__t7.frame))!;
    for (let index = 0; index < 300; index++) {
      if (frame.action_space === 'DISCARD' && frame.projection.player_controls?.some(c => c.effective)) break;
      const prior = frame.revision;
      await page.evaluate(index => {
        const { frame: f, socket } = window.__t7;
        if (!f?.prompt) throw new Error('No pending decision');
        const controls = f.projection.player_controls?.length;
        const offer = f.offers.find(o =>
          (o.action_type === 'PRIORITY_PLAY_LAND' && !controls)
          || (o.action_type === 'PRIORITY_CAST_SPELL' && f.projection.agent.battlefield.length >= 14 && !controls)
          || ['PAY_COST', 'CHOOSE_TARGET'].includes(o.action_type ?? '')
        ) ?? f.offers.find(o => o.action_type === 'PRIORITY_PASS_PRIORITY') ?? f.offers[0];
        socket!.send(JSON.stringify({ type: 'command', command: {
          command_id: `demo.setup.${index}`, match_id: f.match_id,
          expected_revision: f.revision, prompt_id: f.prompt.id, offer_id: offer.id, answers: [],
        } }));
      }, index);
      await expect.poll(() => page.evaluate(() => window.__t7.frame?.revision)).toBeGreaterThan(prior);
      frame = (await page.evaluate(() => window.__t7.frame))!;
    }
    expect(frame.action_space).toBe('DISCARD');
    expect(frame.projection.player_controls?.some(c => c.effective)).toBe(true);
    await expect(page.getByTestId('decision-context')).toContainText('Seat 1 chooses for Seat 2');
    const inspection = page.getByRole('region', { name: 'Private inspection', exact: true });
    await expect(inspection).toContainText("opponent's hand");
    await expect(page.getByTestId('decision-surface')).toHaveAttribute('data-revision', String(frame.revision));
    expect(frame.projection.agent.inspected).toHaveLength(8);
    expect(frame.projection.opponent.hand).toEqual([]);
    const before = { hash: frame.frame_hash, offers: frame.offers, inspected: frame.projection.agent.inspected };
    await page.screenshot({ path: info.outputPath('controlled-before-reload.png'), fullPage: true });
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__t7.frame?.frame_hash)).toBe(before.hash);
    expect(await page.evaluate(() => window.__t7.frame?.offers)).toEqual(before.offers);
    expect(await page.evaluate(() => window.__t7.frame?.projection.agent.inspected)).toEqual(before.inspected);
    await expect(inspection).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    const offer = frame.offers[0];
    const option = page.locator(`[data-testid="action-option"][data-offer-id="${offer.id}"]`);
    await option.focus();
    await page.keyboard.press('Enter');
    await expect.poll(() => page.evaluate(() => window.__t7.frame?.revision)).toBeGreaterThan(frame.revision);
    const after = (await page.evaluate(() => window.__t7.frame))!;
    expect(after.projection.player_controls ?? []).toEqual([]);
    expect(after.projection.agent.inspected ?? []).toEqual([]);
    await expect(inspection).toHaveCount(0);
    await page.screenshot({ path: info.outputPath('expired-after-keyboard-choice.png'), fullPage: true });
    expect(errors).toEqual([]);
    await info.attach('runtime-receipt', { body: JSON.stringify({ width, before, after, errors }, null, 2), contentType: 'application/json' });
  });
}
