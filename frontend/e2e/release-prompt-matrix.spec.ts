import { expect, test, type Page, type TestInfo } from '@playwright/test';

import matrixJson from './release-prompt-matrix.json' with { type: 'json' };

interface PromptPolicy {
  priority_cast_order: Record<string, string[]>;
}

interface PromptScenario {
  id: string;
  hero_deck: string;
  villain_deck: string;
  villain_type: 'random';
  seed: number;
  policy: string;
  max_commands: number;
  expected: {
    winner: 0 | 1;
    turn: number;
    commands: number;
    prompt_counts: Record<string, number>;
  };
}

interface ReleasePromptMatrix {
  schema_version: number;
  action_spaces: {
    reachable: string[];
    terminal: string[];
    excluded: Array<{ family: string }>;
  };
  policies: Record<string, PromptPolicy>;
  scenarios: PromptScenario[];
}

interface RenderedAction {
  description: string;
  disabled: boolean;
  type: string;
}

interface TraceSummary {
  id: string;
}

interface TracePayload {
  config: {
    hero_deck_name: string;
    villain_deck_name: string;
    villain_type: string;
    seed: number;
  };
  end_reason: string;
  winner: number | null;
  final_observation: {
    turn: { turn_number: number };
  };
}

interface ScenarioReceipt {
  id: string;
  seed: number;
  commands: number;
  prompt_counts: Record<string, number>;
  winner: number;
  turn: number;
  trace_id: string;
}

const matrix = matrixJson as ReleasePromptMatrix;
const reachableFamilies = new Set(matrix.action_spaces.reachable);

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

async function installScenarioSeed(page: Page, seed: number): Promise<void> {
  await page.addInitScript((scenarioSeed) => {
    const NativeWebSocket = window.WebSocket;

    class SeededWebSocket extends NativeWebSocket {
      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        let outgoing = data;
        if (typeof data === 'string') {
          try {
            const message = JSON.parse(data) as {
              type?: unknown;
              config?: unknown;
            };
            if (message.type === 'new_game') {
              const config =
                message.config &&
                typeof message.config === 'object' &&
                !Array.isArray(message.config)
                  ? (message.config as Record<string, unknown>)
                  : {};
              outgoing = JSON.stringify({
                ...message,
                config: { ...config, seed: scenarioSeed },
              });
            }
          } catch {
            // The application only emits JSON, but preserve unrelated frames.
          }
        }
        super.send(outgoing);
      }
    }

    Object.defineProperty(window, 'WebSocket', {
      configurable: true,
      value: SeededWebSocket,
      writable: true,
    });
  }, seed);
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function renderedActions(page: Page): Promise<RenderedAction[]> {
  return page.getByTestId('action-option').evaluateAll((buttons) =>
    buttons.map((button) => ({
      description: button.getAttribute('data-action-description') ?? '',
      disabled: (button as HTMLButtonElement).disabled,
      type: button.getAttribute('data-action-type') ?? '',
    })),
  );
}

function chooseAction(
  scenario: PromptScenario,
  family: string,
  actions: RenderedAction[],
): number {
  const policy = matrix.policies[scenario.policy];
  const find = (predicate: (action: RenderedAction) => boolean): number =>
    actions.findIndex(predicate);
  let selected = -1;

  if (family === 'PRIORITY') {
    selected = find((action) => action.type === 'PRIORITY_PLAY_LAND');
    if (selected < 0) {
      for (const card of policy.priority_cast_order[scenario.hero_deck]) {
        selected = find(
          (action) =>
            action.type === 'PRIORITY_CAST_SPELL' &&
            action.description.includes(card),
        );
        if (selected >= 0) {
          break;
        }
      }
    }
    if (selected < 0) {
      selected = find((action) => action.type === 'PRIORITY_ACTIVATE_ABILITY');
    }
    if (selected < 0) {
      selected = find((action) => action.type === 'PRIORITY_PASS_PRIORITY');
    }
  } else if (family === 'DECLARE_ATTACKER') {
    selected = find((action) => action.description.startsWith('Attack with '));
  } else if (family === 'DECLARE_BLOCKER') {
    selected = find((action) => action.description.startsWith('Block '));
  } else if (family === 'CHOOSE_TARGET') {
    for (const target of ['Target Villain', 'Target Hero']) {
      selected = find((action) => action.description === target);
      if (selected >= 0) {
        break;
      }
    }
    if (selected < 0) {
      selected = 0;
    }
  } else if (family === 'SCRY') {
    selected = find((action) => action.type === 'SCRY_KEEP');
  } else if (family === 'LOOK_AND_SELECT' || family === 'DISCARD_THEN_DRAW') {
    selected = find((action) => action.type === 'SELECT_CARD');
    if (selected < 0) {
      selected = 0;
    }
  } else if (family === 'PAY_OR_NOT') {
    selected = find((action) => action.type === 'PAY_COST');
    if (selected < 0) {
      selected = 0;
    }
  } else if (family === 'WATERBEND') {
    selected = find((action) => action.type === 'TAP_FOR_COST');
    if (selected < 0) {
      selected = find((action) => action.type === 'PAY_COST');
    }
  }

  if (selected < 0 || selected >= actions.length) {
    throw new Error(
      `${scenario.id}: policy ${scenario.policy} has no action for ${family}: ` +
        JSON.stringify(actions),
    );
  }
  return selected;
}

async function findTerminalTrace(
  page: Page,
  scenario: PromptScenario,
): Promise<{ id: string; payload: TracePayload }> {
  const summariesResponse = await page.request.get('/api/traces');
  expect(summariesResponse.ok()).toBe(true);
  const summaries = (await summariesResponse.json()) as TraceSummary[];

  for (const summary of summaries) {
    const response = await page.request.get(
      `/api/traces/${encodeURIComponent(summary.id)}`,
    );
    expect(response.ok()).toBe(true);
    const payload = (await response.json()) as TracePayload;
    if (
      payload.config.seed === scenario.seed &&
      payload.config.hero_deck_name === scenario.hero_deck &&
      payload.config.villain_deck_name === scenario.villain_deck
    ) {
      return { id: summary.id, payload };
    }
  }

  throw new Error(`${scenario.id}: terminal trace was not persisted`);
}

async function runScenario(
  page: Page,
  scenario: PromptScenario,
  testInfo: TestInfo,
): Promise<ScenarioReceipt> {
  const consoleErrors = collectConsoleErrors(page);
  await installScenarioSeed(page, scenario.seed);
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });

  await page.getByTestId('deck-select-hero').selectOption(scenario.hero_deck);
  await page
    .getByTestId('deck-select-villain')
    .selectOption(scenario.villain_deck);
  await page.getByTestId('opponent-select').selectOption(scenario.villain_type);
  await page.getByRole('button', { name: 'New Game' }).first().click();

  const expectedDeckNames =
    scenario.hero_deck === 'ur_lessons'
      ? 'UR Lessons vs GW Allies'
      : 'GW Allies vs UR Lessons';
  await expect(page.getByTestId('deck-names')).toHaveText(expectedDeckNames, {
    timeout: 15_000,
  });

  const actionPanel = page.getByTestId('action-panel');
  const actionButtons = page.getByTestId('action-option');
  const gameOver = page.getByText('Game Over', { exact: true });
  const promptCounts: Record<string, number> = {};
  let commands = 0;

  while (commands < scenario.max_commands) {
    await expect(actionButtons.first().or(gameOver)).toBeVisible({
      timeout: 30_000,
    });
    if (await gameOver.isVisible()) {
      break;
    }

    const family = (await actionPanel.getAttribute('data-action-space-kind')) ?? '';
    expect(family, `${scenario.id}: action-space family is absent`).not.toBe('');
    expect(
      reachableFamilies.has(family),
      `${scenario.id}: unexpected action-space family ${family}`,
    ).toBe(true);

    const actions = await renderedActions(page);
    expect(
      actions.length,
      `${scenario.id}: ${family} has no rendered actions`,
    ).toBeGreaterThan(0);
    for (const action of actions) {
      expect(
        action.type,
        `${scenario.id}: ${family} action type is absent`,
      ).not.toBe('');
      expect(
        action.description.trim(),
        `${scenario.id}: ${family} action label is empty`,
      ).not.toBe('');
      expect(
        action.disabled,
        `${scenario.id}: ${family} action is disabled`,
      ).toBe(false);
    }

    promptCounts[family] = (promptCounts[family] ?? 0) + 1;
    const choice = chooseAction(scenario, family, actions);
    const sequenceBefore = await updateSequence(page);
    await actionButtons.nth(choice).click();
    commands += 1;
    await expect
      .poll(() => updateSequence(page), {
        timeout: 30_000,
        message: `${scenario.id}: no authority update after command ${commands}`,
      })
      .toBeGreaterThan(sequenceBefore);
  }

  expect(
    await gameOver.isVisible(),
    `${scenario.id}: game did not reach terminal within ${scenario.max_commands} commands`,
  ).toBe(true);
  expect(commands).toBe(scenario.expected.commands);
  expect(promptCounts).toEqual(scenario.expected.prompt_counts);

  const winnerText = scenario.expected.winner === 0 ? 'Hero wins' : 'Opponent wins';
  await expect(page.getByText(winnerText, { exact: true })).toBeVisible();
  await expect(page.getByTestId('game-board')).toContainText(`Turn ${scenario.expected.turn}`);

  const trace = await findTerminalTrace(page, scenario);
  expect(trace.payload.config.villain_type).toBe(scenario.villain_type);
  expect(trace.payload.end_reason).toBe('game_over');
  expect(trace.payload.winner).toBe(scenario.expected.winner);
  expect(trace.payload.final_observation.turn.turn_number).toBe(
    scenario.expected.turn,
  );
  expect(
    consoleErrors,
    `${scenario.id}: browser errors\n${consoleErrors.join('\n')}`,
  ).toEqual([]);

  const receipt: ScenarioReceipt = {
    id: scenario.id,
    seed: scenario.seed,
    commands,
    prompt_counts: promptCounts,
    winner: trace.payload.winner as number,
    turn: trace.payload.final_observation.turn.turn_number,
    trace_id: trace.id,
  };
  await testInfo.attach(`${scenario.id}.json`, {
    body: Buffer.from(JSON.stringify(receipt, null, 2)),
    contentType: 'application/json',
  });
  return receipt;
}

test('release stack reaches terminal and covers the selected-matchup prompt matrix', async ({
  browser,
}, testInfo) => {
  test.setTimeout(600_000);
  const baseURL = testInfo.project.use.baseURL;
  expect(typeof baseURL).toBe('string');

  const receipts: ScenarioReceipt[] = [];
  const observedFamilies = new Set<string>();
  for (const scenario of matrix.scenarios) {
    const context = await browser.newContext({ baseURL: baseURL as string });
    const page = await context.newPage();
    const receipt = await runScenario(page, scenario, testInfo);
    receipts.push(receipt);
    Object.keys(receipt.prompt_counts).forEach((family) =>
      observedFamilies.add(family),
    );
    await context.close();
  }

  expect([...observedFamilies].sort()).toEqual(
    [...matrix.action_spaces.reachable].sort(),
  );
  await testInfo.attach('release-prompt-matrix.json', {
    body: Buffer.from(
      JSON.stringify({ schema_version: matrix.schema_version, receipts }, null, 2),
    ),
    contentType: 'application/json',
  });
});
