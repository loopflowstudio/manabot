import { expect, test, type Browser, type Page } from '@playwright/test';

import boltFixtureJson from '../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  Command,
  ExperienceFrame,
  FrameUpdate,
  InteractionOffer,
  Observation,
  RecoveryEnvelope,
} from '../src/lib/types';

type InputMode = 'keyboard' | 'pointer';

interface CombatPromptReceipt {
  attackerMode: InputMode;
  blockerMode: InputMode;
  commands: Command[];
}

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };

function combatObservation(): Observation {
  const observation = structuredClone(boltFixture.recovery.frame.projection);
  observation.turn = {
    turn_number: 4,
    phase: 'COMBAT',
    step: 'DECLARE_ATTACKERS_STEP',
    active_player_id: 0,
    agent_player_id: 0,
  };
  observation.agent.battlefield = [{
    id: 83,
    name: 'Otter-Penguin',
    controller_id: 0,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 2,
    toughness: 2,
    base_power: 2,
    base_toughness: 2,
    plus1_counters: 0,
  }];
  observation.opponent.battlefield = [{
    id: 84,
    name: 'Badgermole Cub',
    controller_id: 1,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 1,
    toughness: 1,
    base_power: 1,
    base_toughness: 1,
    plus1_counters: 0,
  }];
  return observation;
}

function offer(
  id: number,
  verb: 'declare_attackers' | 'declare_blockers' | 'pass_priority',
  label: string,
  actionType: string,
  focus: number[],
): InteractionOffer {
  return {
    id,
    actor: 0,
    verb,
    source: null,
    label,
    help: null,
    choices: [],
    confirm_label: label,
    action_type: actionType,
    focus,
  };
}

function frameAt(stage: 'attack' | 'block' | 'priority'): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const revision = stage === 'attack' ? 10 : stage === 'block' ? 11 : 12;
  const promptId = stage === 'attack' ? 110 : stage === 'block' ? 111 : 112;
  const observation = combatObservation();
  const actionSpace = stage === 'attack'
    ? 'DECLARE_ATTACKER'
    : stage === 'block'
      ? 'DECLARE_BLOCKER'
      : 'PRIORITY';
  if (stage === 'block') {
    observation.turn.step = 'DECLARE_BLOCKERS_STEP';
  }
  const offers = stage === 'attack'
    ? [
        offer(101, 'declare_attackers', 'Attack with Otter-Penguin', 'DECLARE_ATTACKER', [83]),
        offer(102, 'declare_attackers', 'Do not attack with Otter-Penguin', 'DECLARE_ATTACKER', [83]),
      ]
    : stage === 'block'
      ? [
          offer(201, 'declare_blockers', 'Block Badgermole Cub with Otter-Penguin', 'DECLARE_BLOCKER', [84, 83]),
          offer(202, 'declare_blockers', 'Otter-Penguin: do not block', 'DECLARE_BLOCKER', [83]),
        ]
      : [offer(301, 'pass_priority', 'Pass priority', 'PRIORITY_PASS_PRIORITY', [])];

  return {
    ...base,
    match_id: 'combat-prompt-proof',
    revision,
    frame_hash: `combat-frame-${revision}`,
    prompt: {
      id: promptId,
      actor: 0,
      kind: actionSpace.toLowerCase(),
      title: 'Choose a combat action',
      instruction: 'Choose a combat action',
    },
    projection: observation,
    offers,
    action_space: actionSpace,
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}

function acceptedUpdate(command: Command, frame: ExperienceFrame): FrameUpdate {
  return {
    base_revision: command.expected_revision,
    frame,
    presentation: [],
    receipt: {
      command_id: command.command_id,
      actor: 0,
      accepted_at: command.expected_revision,
      resulting_revision: frame.revision,
      resulting_frame_hash: frame.frame_hash,
    },
  };
}

async function installCombatAuthority(page: Page, commands: Command[]): Promise<void> {
  await page.routeWebSocket('**/ws/play', (socket) => {
    socket.onMessage((raw) => {
      const message = JSON.parse(String(raw)) as {
        type?: string;
        command?: Command;
      };
      if (message.type === 'new_game') {
        const frame = frameAt('attack');
        socket.send(JSON.stringify({
          type: 'observation',
          data: frame.projection,
          actions: [],
          session_id: 'combat-session',
          resume_token: 'combat-resume-token',
          recovery: {
            ...structuredClone(boltFixture.recovery),
            reason: 'initial_connect',
            frame,
            presentation_tail: [],
            accepted_commands: [],
            replay_cursor: frame.revision,
          },
        }));
        return;
      }
      if (message.type !== 'command' || !message.command) {
        return;
      }

      const command = message.command;
      commands.push(command);
      const next = command.expected_revision === 10 ? frameAt('block') : frameAt('priority');
      socket.send(JSON.stringify({
        type: 'command_outcome',
        status: 'accepted',
        update: acceptedUpdate(command, next),
      }));
    });
  });
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function assertPrompt(
  page: Page,
  family: 'DECLARE_ATTACKER' | 'DECLARE_BLOCKER',
  expectedLabels: string[],
): Promise<void> {
  const panel = page.getByTestId('action-panel');
  const options = page.getByTestId('action-option');
  await expect(panel).toHaveAttribute('data-action-space-kind', family);
  await expect(page.getByTestId('decision-prompt')).toHaveAttribute('data-kind', family);
  await expect(options).toHaveCount(expectedLabels.length);
  await expect(options.first()).toBeFocused();
  for (let index = 0; index < expectedLabels.length; index += 1) {
    await expect(options.nth(index)).toHaveAccessibleName(expectedLabels[index]);
    await expect(options.nth(index)).toBeEnabled();
  }
  await page.keyboard.press('Tab');
  await expect(options.nth(1)).toBeFocused();
  await expect(options.nth(1)).toHaveCSS('outline-style', 'solid');
  await page.keyboard.press('Shift+Tab');
  await expect(options.first()).toBeFocused();
  await expect(options.first()).toHaveCSS('outline-style', 'solid');
}

async function activate(
  page: Page,
  mode: InputMode,
  expected: { offerId: number; promptId: number; revision: number },
  commands: Command[],
): Promise<void> {
  const before = await updateSequence(page);
  const option = page.getByTestId('action-option').first();
  if (mode === 'keyboard') {
    await expect(option).toBeFocused();
    await expect(option).toHaveCSS('outline-style', 'solid');
    await page.keyboard.press('Enter');
  } else {
    await option.click();
  }
  await expect.poll(() => commands.length).toBeGreaterThan(0);
  expect(commands.at(-1)).toMatchObject({
    match_id: 'combat-prompt-proof',
    expected_revision: expected.revision,
    prompt_id: expected.promptId,
    offer_id: expected.offerId,
    answers: [],
  });
  expect(commands.at(-1)?.command_id).toMatch(/\S+/);
  await expect.poll(() => updateSequence(page)).toBeGreaterThan(before);
}

async function runPromptProof(
  browser: Browser,
  attackerMode: InputMode,
  blockerMode: InputMode,
): Promise<CombatPromptReceipt> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const commands: Command[] = [];
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  await installCombatAuthority(page, commands);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'New Game' }).first().click();

  await assertPrompt(page, 'DECLARE_ATTACKER', [
    'Attack with Otter-Penguin',
    'Do not attack with Otter-Penguin',
  ]);
  await activate(page, attackerMode, { offerId: 101, promptId: 110, revision: 10 }, commands);

  await assertPrompt(page, 'DECLARE_BLOCKER', [
    'Block Badgermole Cub with Otter-Penguin',
    'Otter-Penguin: do not block',
  ]);
  await activate(page, blockerMode, { offerId: 201, promptId: 111, revision: 11 }, commands);

  expect(errors).toEqual([]);
  await context.close();
  return { attackerMode, blockerMode, commands };
}

test('combat prompts submit only current offers by pointer and keyboard', async ({ browser }) => {
  const pointerThenKeyboard = await runPromptProof(browser, 'pointer', 'keyboard');
  const keyboardThenPointer = await runPromptProof(browser, 'keyboard', 'pointer');

  expect(pointerThenKeyboard.commands.map((command) => command.offer_id)).toEqual([101, 201]);
  expect(keyboardThenPointer.commands.map((command) => command.offer_id)).toEqual([101, 201]);
});
