import { expect, test, type Page } from '@playwright/test';
import positionsJson from '../../etude/fixtures/choice-navigation.json' with { type: 'json' };
import boltJson from '../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { Command, ExperienceFrame, RecoveryEnvelope } from '../src/lib/types';

const positions = positionsJson.positions;
type Family = keyof typeof positions;

declare global {
  interface Window {
    __choiceNavigation: {
      socket: WebSocket | null;
      frame: ExperienceFrame | null;
      status: string;
    };
  }
}

async function installPosition(page: Page, family: Family): Promise<Command[]> {
  const frame = positions[family].frame as ExperienceFrame;
  const commands: Command[] = [];
  const recovery: RecoveryEnvelope = {
    ...(boltJson.recovery as RecoveryEnvelope),
    frame,
    presentation_tail: [],
    accepted_commands: [],
    replay_cursor: frame.revision,
  };
  await page.routeWebSocket('**/ws/play', socket => {
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw));
      if (message.type === 'new_game') {
        socket.send(JSON.stringify({ type: 'observation', data: frame.projection,
          actions: [], recovery, session_id: 'navigation', resume_token: 'navigation' }));
      } else if (message.type === 'command') {
        commands.push(message.command);
        // Exercise the existing recovery path without claiming mocked acceptance.
        socket.send(JSON.stringify({ type: 'command_outcome', status: 'rejected',
          rejection: { command_id: message.command.command_id, code: 'stale_revision',
            message: 'Recovering the current decision.', current_revision: frame.revision,
            current_prompt: frame.prompt!.id }, recovery }));
      }
    });
  });
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
  await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', family);
  return commands;
}

for (const family of ['CHOOSE_TARGET', 'DISCARD'] as const) {
  test(`${family}: every original offer survives filtering and keyboard recovery`, async ({ page }) => {
    const commands = await installPosition(page, family);
    const offers = positions[family].frame.offers;
    const options = page.getByTestId('action-option');
    const filter = page.getByRole('searchbox', { name: 'Find an action' });
    await expect(options).toHaveCount(offers.length);
    for (const offer of offers) {
      await filter.fill(`  ${offer.label.toUpperCase()}  `);
      const matching = offers.filter(candidate => candidate.label.toLowerCase().includes(offer.label.toLowerCase()));
      await expect(options).toHaveCount(matching.length);
      expect(await options.evaluateAll(buttons => buttons.map(button => Number(button.getAttribute('data-offer-id')))))
        .toEqual(matching.map(candidate => candidate.id));
    }
    expect(commands).toEqual([]);
    await filter.fill('no such card');
    await expect(options).toHaveCount(0);
    await expect(page.getByText('No matching actions. Clear the filter to see every choice.')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(filter).toHaveValue('');
    await expect(filter).toBeFocused();
    await expect(options).toHaveCount(offers.length);
    const choice = offers[0];
    await filter.fill(choice.label);
    await page.getByRole('button', { name: 'Clear filter', exact: true }).click();
    await expect(filter).toBeFocused();
    await expect(options).toHaveCount(offers.length);
    await filter.fill(choice.label);
    await page.keyboard.press('Tab'); // Clear filter
    await page.keyboard.press('Tab'); // first matching offer
    await expect(options.first()).toBeFocused();
    await page.keyboard.press('Enter');
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toMatchObject({ offer_id: choice.id,
      expected_revision: positions[family].frame.revision,
      prompt_id: positions[family].frame.prompt!.id,
      match_id: positions[family].frame.match_id, answers: [] });
    await expect(filter).toHaveValue('');
    await expect(options).toHaveCount(offers.length);
    await expect(options.first()).toBeFocused();
  });
}

test('small optional and combat decisions retain every direct choice', async ({ page }) => {
  for (const family of ['PAY_OR_NOT', 'DECLARE_BLOCKER'] as const) {
    const commands = await installPosition(page, family);
    await expect(page.getByRole('searchbox', { name: 'Find an action' })).toHaveCount(0);
    await expect(page.getByTestId('action-option')).toHaveCount(positions[family].frame.offers.length);
    expect(commands).toEqual([]);
    await page.unrouteAll();
  }
});

test('retained large Learn decision keeps modes, cancellation and all original copies', async ({ page }) => {
  const commands = await installPosition(page, 'LEARN');
  for (const [label, type] of [['Discard and draw', 'LEARN_DISCARD'], ['Take a Lesson', 'LEARN_TAKE_LESSON']]) {
    await page.getByRole('button', { name: label, exact: true }).click();
    await expect(page.getByTestId('action-option')).toHaveCount(
      positions.LEARN.frame.offers.filter(offer => offer.action_type === type).length,
    );
    await page.getByRole('button', { name: 'Back', exact: true }).click();
  }
  await expect(page.getByRole('button', { name: 'Decline Learn', exact: true })).toBeVisible();
  expect(commands).toEqual([]);
});

// The review path uses the real backend, the retained legal prefix and ordinary
// Commands. Set ETUDE_REVIEW_CHOICE to a family and run headed to pause for play.
for (const family of ['CHOOSE_TARGET', 'DISCARD'] as const) {
  test(`live retained ${family} can execute an intended choice`, async ({ page }) => {
    test.skip(Boolean(process.env.ETUDE_REVIEW_CHOICE && process.env.ETUDE_REVIEW_CHOICE !== family));
    const position = positions[family];
    await page.addInitScript(config => {
      const state: Window['__choiceNavigation'] = {
        socket: null, frame: null, status: '',
      };
      const Native = window.WebSocket;
      class TrackedSocket extends Native {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          state.socket = this;
          this.addEventListener('message', event => {
            const message = JSON.parse(event.data);
            if (message.type === 'command_outcome') state.status = message.status;
            const findFrame = (value: unknown): void => {
              if (!value || typeof value !== 'object') return;
              if ('frame_hash' in value && 'projection' in value) state.frame = value as ExperienceFrame;
              else for (const child of Object.values(value)) findFrame(child);
            };
            findFrame(message);
          });
        }
        send(data: string): void {
          const message = JSON.parse(data);
          if (message.type === 'new_game') message.config = config;
          super.send(JSON.stringify(message));
        }
      }
      window.__choiceNavigation = state;
      window.WebSocket = TrackedSocket;
    }, position.config);
    await page.goto('/');
    await expect(page.getByTestId('connection-badge')).toHaveText('connected');
    await page.getByRole('button', { name: 'New Game', exact: true }).first().click();
    await expect(page.getByTestId('action-option').first()).toBeVisible();
    for (const [ordinal, step] of position.prefix.entries()) {
      const revision = await page.evaluate(({ ordinal, step }) => {
        const state = window.__choiceNavigation;
        const frame = state.frame!;
        if (frame.offers.find(offer => offer.id === step.offer_id)?.label !== step.label) {
          throw new Error(`Retained prefix diverged at ${ordinal}`);
        }
        state.socket!.send(JSON.stringify({ type: 'command', command: {
          command_id: `navigation.live.${ordinal}`, match_id: frame.match_id,
          expected_revision: frame.revision, prompt_id: frame.prompt!.id,
          offer_id: step.offer_id, answers: [],
        } }));
        return frame.revision;
      }, { ordinal, step });
      await expect.poll(() => page.evaluate(() =>
        window.__choiceNavigation.frame!.revision,
      )).toBeGreaterThan(revision);
    }
    await expect(page.getByTestId('action-panel')).toHaveAttribute('data-action-space-kind', family);
    const actual = await page.evaluate(() =>
      window.__choiceNavigation.frame!,
    );
    expect(actual.offers).toEqual(position.frame.offers);
    expect(actual.projection).toEqual(position.frame.projection);
    if (process.env.ETUDE_REVIEW_CHOICE === family) {
      await page.pause();
      return;
    }
    const intended = family === 'CHOOSE_TARGET' ? 'Target Earth Kingdom Jailer' : position.frame.offers[0].label;
    await page.getByRole('searchbox', { name: 'Find an action' }).fill(intended);
    await page.getByRole('button', { name: intended, exact: true }).first().click();
    await expect.poll(() => page.evaluate(() =>
      window.__choiceNavigation.frame!.revision,
    )).toBeGreaterThan(actual.revision);
    expect(await page.evaluate(() =>
      window.__choiceNavigation.status,
    )).toBe('accepted');
  });
}
