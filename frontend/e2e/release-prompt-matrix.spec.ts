import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page, type TestInfo } from '@playwright/test';

import { DECISION_PROMPTS } from '../src/lib/prompt-instructions';
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
  offer_id: number;
  type: string;
}

interface MatrixCommand {
  command_id: string;
  expected_revision: number;
  offer_id: number;
  prompt_id: number;
}

interface MatrixBrowserState {
  commands: MatrixCommand[];
  connectionObserver?: MutationObserver;
  connectionStatuses: string[];
  sockets: WebSocket[];
}

type MatrixWindow = Window & typeof globalThis & {
  __manabotMatrix?: MatrixBrowserState;
};

interface RuntimeFailures {
  console: string[];
  localResponses: string[];
  publicRequests: string[];
  requestFailures: string[];
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
  accessibility: {
    audited_families: string[];
    keyboard_commands: number;
    reconnect_statuses: string[];
    reduced_motion: true;
  };
}

const matrix = matrixJson as ReleasePromptMatrix;
const reachableFamilies = new Set(matrix.action_spaces.reachable);
const PUBLIC_PROTOCOLS = new Set(['http:', 'https:', 'ws:', 'wss:']);
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '[::1]']);
const AXE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

function isPublicRequest(rawUrl: string): boolean {
  const url = new URL(rawUrl);
  return PUBLIC_PROTOCOLS.has(url.protocol) && !LOOPBACK_HOSTS.has(url.hostname);
}

function collectRuntimeFailures(page: Page): RuntimeFailures {
  const failures: RuntimeFailures = {
    console: [],
    localResponses: [],
    publicRequests: [],
    requestFailures: [],
  };

  page.on('console', (message) => {
    if (message.type() === 'error') {
      failures.console.push(message.text());
    }
  });
  page.on('pageerror', (error) => failures.console.push(`pageerror: ${error.message}`));
  page.on('request', (request) => {
    if (isPublicRequest(request.url())) {
      failures.publicRequests.push(request.url());
    }
  });
  page.on('requestfailed', (request) => {
    failures.requestFailures.push(
      `${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'unknown failure'}`,
    );
  });
  page.on('response', (response) => {
    if (!isPublicRequest(response.url()) && response.status() >= 400) {
      failures.localResponses.push(`${response.status()} ${response.url()}`);
    }
  });
  return failures;
}

function assertNoRuntimeFailures(scenario: PromptScenario, failures: RuntimeFailures): void {
  expect(
    failures.console,
    `${scenario.id}: browser errors\n${failures.console.join('\n')}`,
  ).toEqual([]);
  expect(
    failures.localResponses,
    `${scenario.id}: broken local responses\n${failures.localResponses.join('\n')}`,
  ).toEqual([]);
  expect(
    failures.requestFailures,
    `${scenario.id}: failed requests\n${failures.requestFailures.join('\n')}`,
  ).toEqual([]);
  expect(
    failures.publicRequests,
    `${scenario.id}: public play assets\n${failures.publicRequests.join('\n')}`,
  ).toEqual([]);
}

async function installScenarioInstrumentation(page: Page, seed: number): Promise<void> {
  await page.addInitScript((scenarioSeed) => {
    const NativeWebSocket = window.WebSocket;
    const state: MatrixBrowserState = {
      commands: [],
      connectionStatuses: [],
      sockets: [],
    };

    class SeededWebSocket extends NativeWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        if (protocols === undefined) {
          super(url);
        } else {
          super(url, protocols);
        }
        state.sockets.push(this);
      }

      send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        let outgoing = data;
        if (typeof data === 'string') {
          try {
            const message = JSON.parse(data) as {
              type?: unknown;
              config?: unknown;
              command?: unknown;
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
            } else if (
              message.type === 'command' &&
              message.command &&
              typeof message.command === 'object' &&
              !Array.isArray(message.command)
            ) {
              const command = message.command as Record<string, unknown>;
              state.commands.push({
                command_id: String(command.command_id ?? ''),
                expected_revision: Number(command.expected_revision),
                offer_id: Number(command.offer_id),
                prompt_id: Number(command.prompt_id),
              });
            }
          } catch {
            // The application emits JSON; preserve unrelated frames unchanged.
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
    (window as MatrixWindow).__manabotMatrix = state;
  }, seed);
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function commandCount(page: Page): Promise<number> {
  return page.evaluate(() => (window as MatrixWindow).__manabotMatrix?.commands.length ?? 0);
}

async function renderedActions(page: Page): Promise<RenderedAction[]> {
  return page.getByTestId('action-option').evaluateAll((buttons) =>
    buttons.map((button) => ({
      description: button.getAttribute('data-action-description') ?? '',
      disabled: (button as HTMLButtonElement).disabled,
      offer_id: Number(button.getAttribute('data-offer-id')),
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

async function assertAccessiblePrompt(
  page: Page,
  scenario: PromptScenario,
  family: string,
  actions: RenderedAction[],
): Promise<void> {
  const instruction = DECISION_PROMPTS[family];
  expect(instruction, `${scenario.id}: ${family} has no prompt instruction`).toBeTruthy();

  const panel = page.getByTestId('action-panel');
  const prompt = page.getByTestId('decision-prompt');
  const actionButtons = page.getByTestId('action-option');
  await expect(panel).toHaveAccessibleName('Actions');
  await expect(panel).toHaveAccessibleDescription(instruction);
  await expect(prompt).toHaveText(instruction);
  await expect(panel.getByRole('status')).toHaveText('Your move');
  await expect(actionButtons).toHaveCount(actions.length);

  for (let index = 0; index < actions.length; index += 1) {
    const action = actions[index];
    const button = actionButtons.nth(index);
    expect(action.offer_id, `${scenario.id}: ${family} offer id is absent`).toBeGreaterThanOrEqual(0);
    expect(action.type, `${scenario.id}: ${family} action type is absent`).not.toBe('');
    expect(action.description.trim(), `${scenario.id}: ${family} action label is empty`).not.toBe('');
    expect(action.disabled, `${scenario.id}: ${family} action is disabled`).toBe(false);
    await expect(button).toHaveAccessibleName(action.description);
    await expect(button).toHaveAccessibleDescription(instruction);
  }
  await expect(actionButtons.first()).toBeFocused();
}

async function assertFocusBoundary(page: Page, actionCount: number): Promise<void> {
  const panel = page.getByTestId('action-panel');
  const actionButtons = page.getByTestId('action-option');
  await expect(actionButtons.first()).toBeFocused();

  for (let index = 1; index < actionCount; index += 1) {
    await page.keyboard.press('Tab');
    await expect(actionButtons.nth(index)).toBeFocused();
  }

  await page.keyboard.press('Tab');
  expect(
    await panel.evaluate((element) => element.contains(document.activeElement)),
    'focus was trapped in the Actions region',
  ).toBe(false);
  expect(
    await page.evaluate(() => document.activeElement === null || document.activeElement === document.body),
    'focus was lost after the final legal choice',
  ).toBe(false);

  await page.keyboard.press('Shift+Tab');
  await expect(actionButtons.last()).toBeFocused();
  for (let index = actionCount - 2; index >= 0; index -= 1) {
    await page.keyboard.press('Shift+Tab');
    await expect(actionButtons.nth(index)).toBeFocused();
  }
}

async function assertReducedMotion(page: Page, label: string): Promise<void> {
  expect(
    await page.evaluate(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches),
    `${label}: reduced-motion media query is not active`,
  ).toBe(true);

  const offenders = await page.locator('body *:visible').evaluateAll((elements) => {
    const maxMilliseconds = (value: string): number =>
      Math.max(
        0,
        ...value.split(',').map((part) => {
          const token = part.trim();
          if (token.endsWith('ms')) {
            return Number.parseFloat(token);
          }
          if (token.endsWith('s')) {
            return Number.parseFloat(token) * 1000;
          }
          return 0;
        }),
      );

    return elements.flatMap((element) => {
      const style = window.getComputedStyle(element);
      const animationMs = maxMilliseconds(style.animationDuration);
      const transitionMs = maxMilliseconds(style.transitionDuration);
      if (animationMs <= 1 && transitionMs <= 1) {
        return [];
      }
      const identity =
        element.getAttribute('data-testid') ??
        element.getAttribute('aria-label') ??
        element.tagName.toLowerCase();
      return [`${identity}: animation=${animationMs}ms transition=${transitionMs}ms`];
    });
  });
  expect(offenders, `${label}: perceptible motion remains`).toEqual([]);

  const stage = page.getByTestId('presentation-stage');
  if (await stage.isVisible()) {
    await expect(stage).toHaveAttribute('data-reduced-motion', 'true');
  }
}

async function auditAccessibility(page: Page, label: string): Promise<void> {
  const results = await new AxeBuilder({ page }).withTags(AXE_TAGS).analyze();
  const violations = results.violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    help: violation.help,
    targets: violation.nodes.map((node) => node.target.join(' ')),
  }));
  expect(violations, `${label}: WCAG/contrast violations`).toEqual([]);
}

async function assertCuratedAssets(page: Page, label: string): Promise<void> {
  const board = page.getByTestId('game-board');
  const treatments = board.getByTestId('card-treatment');
  const count = await treatments.count();
  expect(count, `${label}: no curated card treatments rendered`).toBeGreaterThan(0);
  await expect(board.locator('[data-asset-source="pack"]')).toHaveCount(count);
  await expect(board.locator('[data-asset-source="fallback"]')).toHaveCount(0);
}

async function assertExistingReconnectStatus(page: Page): Promise<string[]> {
  const badge = page.getByTestId('connection-badge');
  const actionsBefore = await renderedActions(page);
  const sequenceBefore = await updateSequence(page);

  await page.evaluate(() => {
    const matrixState = (window as MatrixWindow).__manabotMatrix;
    const connectionBadge = document.querySelector<HTMLElement>('[data-testid="connection-badge"]');
    if (!matrixState || !connectionBadge) {
      throw new Error('matrix reconnect instrumentation is unavailable');
    }
    matrixState.connectionStatuses = [];
    matrixState.connectionObserver?.disconnect();
    const record = (value: string | null): void => {
      if (value && matrixState.connectionStatuses.at(-1) !== value) {
        matrixState.connectionStatuses.push(value);
      }
    };
    record(connectionBadge.getAttribute('data-connection-state'));
    matrixState.connectionObserver = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        record(mutation.oldValue);
        record(connectionBadge.getAttribute('data-connection-state'));
      }
    });
    matrixState.connectionObserver.observe(connectionBadge, {
      attributeFilter: ['data-connection-state'],
      attributeOldValue: true,
      attributes: true,
    });

    const socket = matrixState.sockets.at(-1);
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error('no open WebSocket is available for reconnect status proof');
    }
    socket.close(4102, 'release accessibility reconnect status proof');
  });

  await expect(badge).toHaveAttribute('data-connection-state', 'disconnected', {
    timeout: 5_000,
  });
  await expect(badge).toHaveAttribute('data-connection-state', 'connected', {
    timeout: 15_000,
  });
  await expect(badge).toHaveAccessibleName('Connection status: connected');
  await expect
    .poll(() => updateSequence(page), {
      timeout: 15_000,
      message: 'resume did not restore an authoritative frame',
    })
    .toBeGreaterThan(sequenceBefore);
  expect(await renderedActions(page)).toEqual(actionsBefore);
  await expect(page.getByTestId('action-option').first()).toBeFocused();

  const statuses = await page.evaluate(
    () => (window as MatrixWindow).__manabotMatrix?.connectionStatuses ?? [],
  );
  expect(statuses).toEqual(expect.arrayContaining(['disconnected', 'reconnecting', 'connected']));
  return statuses;
}

async function activateKeyboardChoice(
  page: Page,
  scenario: PromptScenario,
  family: string,
  actions: RenderedAction[],
  choice: number,
  commandNumber: number,
): Promise<void> {
  const actionButtons = page.getByTestId('action-option');
  for (let index = 0; index < choice; index += 1) {
    await page.keyboard.press('Tab');
  }
  await expect(actionButtons.nth(choice)).toBeFocused();

  const legalOfferIds = actions.map((action) => action.offer_id);
  const selectedOfferId = actions[choice].offer_id;
  const commandsBefore = await commandCount(page);
  const sequenceBefore = await updateSequence(page);
  await page.keyboard.press(commandNumber % 2 === 0 ? 'Enter' : 'Space');

  await expect
    .poll(() => commandCount(page), {
      timeout: 5_000,
      message: `${scenario.id}: ${family} keyboard activation sent no command`,
    })
    .toBe(commandsBefore + 1);
  const command = await page.evaluate(
    () => (window as MatrixWindow).__manabotMatrix?.commands.at(-1) ?? null,
  );
  expect(command, `${scenario.id}: ${family} command was not captured`).not.toBeNull();
  expect(command?.command_id, `${scenario.id}: ${family} command id is absent`).not.toBe('');
  expect(legalOfferIds).toContain(command?.offer_id);
  expect(command?.offer_id).toBe(selectedOfferId);
  expect(command?.expected_revision).toBeGreaterThanOrEqual(0);
  expect(command?.prompt_id).toBeGreaterThanOrEqual(0);

  await expect
    .poll(() => updateSequence(page), {
      timeout: 30_000,
      message: `${scenario.id}: no authority update after command ${commandNumber + 1}`,
    })
    .toBeGreaterThan(sequenceBefore);
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
  auditedFamilies: Set<string>,
): Promise<ScenarioReceipt> {
  const failures = collectRuntimeFailures(page);
  await installScenarioInstrumentation(page, scenario.seed);
  await page.goto('/');

  const connectionBadge = page.getByTestId('connection-badge');
  await expect(connectionBadge).toHaveText('connected', { timeout: 15_000 });
  await expect(connectionBadge).toHaveAccessibleName('Connection status: connected');
  await page.getByTestId('deck-select-hero').selectOption(scenario.hero_deck);
  await page.getByTestId('deck-select-villain').selectOption(scenario.villain_deck);
  await page.getByTestId('opponent-select').selectOption(scenario.villain_type);
  const newGame = page.getByRole('button', { name: 'New Game' }).first();
  await newGame.focus();
  await expect(newGame).toBeFocused();
  await page.keyboard.press('Enter');

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
  const scenarioAudits: string[] = [];
  let reconnectStatuses: string[] = [];
  let commands = 0;

  while (commands < scenario.max_commands) {
    await expect(actionButtons.first().or(gameOver)).toBeVisible({ timeout: 30_000 });
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
    expect(actions.length, `${scenario.id}: ${family} has no rendered actions`).toBeGreaterThan(0);
    await assertAccessiblePrompt(page, scenario, family, actions);
    await assertCuratedAssets(page, `${scenario.id}: ${family}`);

    if (scenario.id === 'ur-lessons-seed-51' && commands === 0) {
      reconnectStatuses = await assertExistingReconnectStatus(page);
      await assertAccessiblePrompt(page, scenario, family, actions);
      await auditAccessibility(page, `${scenario.id}: reconnected`);
    }

    if (!auditedFamilies.has(family)) {
      await assertFocusBoundary(page, actions.length);
      await assertReducedMotion(page, `${scenario.id}: ${family}`);
      await auditAccessibility(page, `${scenario.id}: ${family}`);
      auditedFamilies.add(family);
      scenarioAudits.push(family);
    }

    promptCounts[family] = (promptCounts[family] ?? 0) + 1;
    const choice = chooseAction(scenario, family, actions);
    await activateKeyboardChoice(page, scenario, family, actions, choice, commands);
    commands += 1;
  }

  expect(
    await gameOver.isVisible(),
    `${scenario.id}: game did not reach terminal within ${scenario.max_commands} commands`,
  ).toBe(true);
  expect(commands).toBe(scenario.expected.commands);
  expect(promptCounts).toEqual(scenario.expected.prompt_counts);

  const winnerText = scenario.expected.winner === 0 ? 'Hero wins' : 'Opponent wins';
  const resultDialog = page.getByTestId('game-result-dialog');
  const resultAction = page.getByTestId('game-result-action');
  await expect(resultDialog).toHaveAccessibleName('Game Over');
  await expect(resultDialog).toHaveAccessibleDescription(winnerText);
  await expect(page.getByTestId('game-result')).toHaveText(winnerText);
  await expect(resultAction).toHaveAccessibleName('Play Again');
  await expect(resultAction).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(resultAction).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(resultAction).toBeFocused();
  await expect(actionPanel.getByRole('status')).toHaveText('Game over');
  await expect(page.getByTestId('game-board')).toContainText(`Turn ${scenario.expected.turn}`);
  await assertReducedMotion(page, `${scenario.id}: GAME_OVER`);
  await auditAccessibility(page, `${scenario.id}: GAME_OVER`);
  await assertCuratedAssets(page, `${scenario.id}: GAME_OVER`);

  const trace = await findTerminalTrace(page, scenario);
  expect(trace.payload.config.villain_type).toBe(scenario.villain_type);
  expect(trace.payload.end_reason).toBe('game_over');
  expect(trace.payload.winner).toBe(scenario.expected.winner);
  expect(trace.payload.final_observation.turn.turn_number).toBe(scenario.expected.turn);
  assertNoRuntimeFailures(scenario, failures);

  const receipt: ScenarioReceipt = {
    id: scenario.id,
    seed: scenario.seed,
    commands,
    prompt_counts: promptCounts,
    winner: trace.payload.winner as number,
    turn: trace.payload.final_observation.turn.turn_number,
    trace_id: trace.id,
    accessibility: {
      audited_families: scenarioAudits,
      keyboard_commands: commands,
      reconnect_statuses: reconnectStatuses,
      reduced_motion: true,
    },
  };
  await testInfo.attach(`${scenario.id}.json`, {
    body: Buffer.from(JSON.stringify(receipt, null, 2)),
    contentType: 'application/json',
  });
  return receipt;
}

test('release stack proves keyboard and reduced-motion accessibility across the prompt matrix', async ({
  browser,
}, testInfo) => {
  test.setTimeout(600_000);
  const baseURL = testInfo.project.use.baseURL;
  expect(typeof baseURL).toBe('string');

  const receipts: ScenarioReceipt[] = [];
  const observedFamilies = new Set<string>();
  const auditedFamilies = new Set<string>();
  for (const scenario of matrix.scenarios) {
    const context = await browser.newContext({
      baseURL: baseURL as string,
      reducedMotion: 'reduce',
    });
    const page = await context.newPage();
    const receipt = await runScenario(page, scenario, testInfo, auditedFamilies);
    receipts.push(receipt);
    Object.keys(receipt.prompt_counts).forEach((family) => observedFamilies.add(family));
    await context.close();
  }

  const expectedFamilies = [...matrix.action_spaces.reachable].sort();
  expect([...observedFamilies].sort()).toEqual(expectedFamilies);
  expect([...auditedFamilies].sort()).toEqual(expectedFamilies);
  await testInfo.attach('release-prompt-matrix.json', {
    body: Buffer.from(
      JSON.stringify(
        {
          schema_version: matrix.schema_version,
          accessibility: {
            audited_families: [...auditedFamilies].sort(),
            reduced_motion: true,
          },
          receipts,
        },
        null,
        2,
      ),
    ),
    contentType: 'application/json',
  });
});
