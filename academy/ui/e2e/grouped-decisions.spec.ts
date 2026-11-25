import { expect, test, type Page } from '@playwright/test';

import boltFixtureJson from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type {
  GroupCommand,
  GroupPlan,
  LiveGroup,
} from '../src/lib/group';
import type {
  CardState,
  Command,
  ExperienceFrame,
  InteractionOffer,
  Observation,
  RecoveryEnvelope,
} from '../src/lib/types';
import { tableSnapshot } from './fixtures/table';

const boltFixture = boltFixtureJson as { recovery: RecoveryEnvelope };
const rid = (entity: number) => ({ entity, incarnation: 0 });

function creature(id: number, name: string): Observation['agent']['battlefield'][number] {
  return {
    id, name, controller_id: 0, tapped: false, damage: 0, summoning_sick: false,
    power: 2, toughness: 2, base_power: 2, base_toughness: 2, plus1_counters: 0,
  };
}

function inspectedCard(id: number, name: string): CardState {
  return {
    id, registry_key: id, name, zone: 'LIBRARY', owner_id: 0, power: 0, toughness: 0, mana_value: 1,
    types: { is_creature: false, is_land: false, is_spell: true, is_artifact: false, is_enchantment: false, is_planeswalker: false, is_battle: false },
  };
}

function offer(id: number, label: string, actionType: string, focus: number[]): InteractionOffer {
  return {
    id, actor: 0, verb: 'choose', source: null, label, help: null, choices: [], confirm_label: label,
    action_type: actionType, focus,
  };
}

function makeFrame(
  revision: number,
  actionSpace: string,
  title: string,
  offers: InteractionOffer[],
  patch?: (obs: Observation) => void,
): ExperienceFrame {
  const base = structuredClone(boltFixture.recovery.frame);
  const projection = structuredClone(base.projection);
  projection.turn = { turn_number: 3, phase: 'PRECOMBAT_MAIN', step: 'PRECOMBAT_MAIN_STEP', active_player_id: 0, agent_player_id: 0 };
  projection.agent.battlefield = [];
  projection.opponent.battlefield = [];
  patch?.(projection);
  return {
    ...base,
    match_id: 'grouped-decisions-proof',
    revision,
    frame_hash: `group-frame-${revision}`,
    prompt: { id: 200 + revision, actor: 0, kind: actionSpace.toLowerCase(), title, instruction: title },
    projection,
    offers,
    action_space: actionSpace,
    deck_names: { hero: 'UR Lessons', villain: 'GW Allies' },
  };
}

interface MockState {
  frame: ExperienceFrame;
  group: LiveGroup | null;
}

interface RecordedMessages {
  commands: Command[];
  groupCommands: GroupCommand[];
  previews: GroupPlan[];
}

async function mountScenario(
  page: Page,
  initial: MockState,
  handlers: {
    onCommand?: (command: Command, current: MockState) => MockState;
    onGroupCommand?: (command: GroupCommand, current: MockState) => MockState;
  },
): Promise<RecordedMessages> {
  const recorded: RecordedMessages = { commands: [], groupCommands: [], previews: [] };
  let state = initial;

  await page.route('**/api/sessions/*/group-preview', async route => {
    const request = route.request().postDataJSON() as {
      match_id: string; revision: number; prompt_id: number; frame_hash: string; plan: GroupPlan;
    };
    recorded.previews.push(request.plan);
    const outcome = (() => {
      switch (request.plan.kind) {
        case 'declare_attackers':
          return { kind: 'declare_attackers', attackers: request.plan.attackers };
        case 'declare_blockers':
          return { kind: 'declare_blockers', blocks: request.plan.blocks };
        case 'scry':
          return { kind: 'scry', top: request.plan.top, bottom: request.plan.bottom };
        case 'surveil': {
          const gy = new Set(request.plan.graveyard.map(item => item.entity));
          const kept = state.group?.kind === 'surveil'
            ? state.group.items.filter(card => !gy.has(card.item.entity)).map(card => card.item)
            : [];
          return { kind: 'surveil', graveyard: request.plan.graveyard, kept };
        }
        case 'order_library':
          return {
            kind: 'order_library',
            end: state.group?.kind === 'order_library' ? state.group.end : 'top',
            order: request.plan.order,
          };
      }
    })();
    await route.fulfill({
      json: {
        match_id: request.match_id,
        revision: request.revision,
        prompt_id: request.prompt_id,
        frame_hash: request.frame_hash,
        viewer: 0,
        complete: true,
        refused: null,
        outcome,
      },
    });
  });

  await page.routeWebSocket('**/ws/play', socket => {
    const recovery = (frame: ExperienceFrame): RecoveryEnvelope => ({
      ...structuredClone(boltFixture.recovery),
      reason: 'initial_connect',
      frame,
      presentation_tail: [],
      presentation_cursor: 0,
      accepted_commands: [],
      replay_cursor: frame.revision,
    });
    socket.onMessage(raw => {
      const message = JSON.parse(String(raw)) as { type?: string; command?: Command | GroupCommand };
      if (message.type === 'new_game' || message.type === 'resume') {
        socket.send(JSON.stringify({
          type: 'observation',
          table: tableSnapshot(),
          data: state.frame.projection,
          actions: [],
          group: state.group,
          session_id: 'grouped-session',
          resume_token: 'grouped-resume-token',
          recovery: recovery(state.frame),
        }));
        return;
      }
      if (message.type === 'group_command' && message.command) {
        const command = message.command as GroupCommand;
        recorded.groupCommands.push(command);
        state = handlers.onGroupCommand ? handlers.onGroupCommand(command, state) : state;
        socket.send(JSON.stringify({
          type: 'command_outcome',
          status: 'accepted',
          group: state.group,
          update: {
            base_revision: command.expected_revision,
            frame: state.frame,
            presentation: [],
            receipt: {
              command_id: command.command_id,
              actor: 0,
              accepted_at: command.expected_revision,
              resulting_revision: state.frame.revision,
              resulting_frame_hash: state.frame.frame_hash,
            },
          },
        }));
        return;
      }
      if (message.type === 'command' && message.command) {
        const command = message.command as Command;
        recorded.commands.push(command);
        state = handlers.onCommand ? handlers.onCommand(command, state) : state;
        socket.send(JSON.stringify({
          type: 'command_outcome',
          status: 'accepted',
          group: state.group,
          update: {
            base_revision: command.expected_revision,
            frame: state.frame,
            presentation: [],
            receipt: {
              command_id: command.command_id,
              actor: 0,
              accepted_at: command.expected_revision,
              resulting_revision: state.frame.revision,
              resulting_frame_hash: state.frame.frame_hash,
            },
          },
        }));
      }
    });
  });

  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game' }).first().click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
  return recorded;
}

for (const count of [3, 1] as const) {
  test(`drafting ${count} attacker${count > 1 ? 's' : ''} and confirming sends one group_command`, async ({ page }) => {
    const ids = Array.from({ length: count }, (_, i) => 50 + i);
    const frame = makeFrame(
      10,
      'DECLARE_ATTACKER',
      'Declare attackers',
      [
        { id: 0, actor: 0, verb: 'declare_attackers', source: null, label: 'Attack with Ogre', help: null, choices: [], confirm_label: 'Attack', action_type: 'DECLARE_ATTACKER', focus: [ids[0]] },
        { id: 1, actor: 0, verb: 'declare_attackers', source: null, label: 'Do not attack', help: null, choices: [], confirm_label: 'No', action_type: 'NO_ATTACK', focus: [ids[0]] },
      ],
      obs => {
        obs.turn = { ...obs.turn, phase: 'COMBAT', step: 'DECLARE_ATTACKERS_STEP' };
        obs.agent.battlefield = ids.map((id, i) => creature(id, `Ogre ${i + 1}`));
      },
    );
    const group: LiveGroup = {
      match_id: frame.match_id,
      revision: frame.revision,
      prompt_id: frame.prompt!.id,
      frame_hash: frame.frame_hash,
      kind: 'declare_attackers',
      declarable: ids.map((id, i) => ({ item: rid(id), label: `Ogre ${i + 1}`, must_attack: false })),
      committed: [],
    };
    const nextFrame = makeFrame(10 + count, 'PRIORITY', 'Your priority', [
      { id: 0, actor: 0, verb: 'pass_priority', source: null, label: 'Pass priority', help: null, choices: [], confirm_label: 'Pass', action_type: 'PRIORITY_PASS_PRIORITY', focus: [] },
    ]);

    const recorded = await mountScenario(page, { frame, group }, {
      onGroupCommand: () => ({ frame: nextFrame, group: null }),
    });

    await expect(page.getByTestId('combat-draft')).toHaveAttribute('data-kind', 'attackers');
    for (const id of ids) {
      await page.locator(`.board [data-testid=board-card][data-object-id="${id}"]`).click();
    }
    const confirm = page.getByTestId('primary-action');
    await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
    await confirm.click();

    await expect.poll(() => recorded.groupCommands.length).toBe(1);
    expect(recorded.commands).toEqual([]);
    expect(recorded.groupCommands[0].plan).toEqual({
      kind: 'declare_attackers',
      attackers: ids.map(rid),
    });
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
  });
}

test('Scry 2 opens ArrangePanel, requires placing both cards, and confirms in one group_command', async ({ page }) => {
  const frame = makeFrame(
    20,
    'SCRY',
    'Scry 2',
    [
      offer(0, 'Put Opt on top', 'SCRY_TOP', [71]),
      offer(1, 'Put Opt on bottom', 'SCRY_BOTTOM', [71]),
      offer(2, 'Put Island on top', 'SCRY_TOP', [72]),
      offer(3, 'Put Island on bottom', 'SCRY_BOTTOM', [72]),
    ],
    obs => {
      obs.agent.inspected = [inspectedCard(71, 'Opt'), inspectedCard(72, 'Island')];
    },
  );
  const group: LiveGroup = {
    match_id: frame.match_id,
    revision: frame.revision,
    prompt_id: frame.prompt!.id,
    frame_hash: frame.frame_hash,
    kind: 'scry',
    items: [{ item: rid(71), label: 'Opt' }, { item: rid(72), label: 'Island' }],
    committed_top: [],
    committed_bottom: [],
  };
  const after = makeFrame(22, 'PRIORITY', 'Your priority', [
    { id: 0, actor: 0, verb: 'pass_priority', source: null, label: 'Pass priority', help: null, choices: [], confirm_label: 'Pass', action_type: 'PRIORITY_PASS_PRIORITY', focus: [] },
  ]);

  const recorded = await mountScenario(page, { frame, group }, {
    onGroupCommand: () => ({ frame: after, group: null }),
  });

  const panel = page.getByTestId('arrange-panel');
  await expect(panel).toBeVisible();
  await expect(panel).toHaveAttribute('data-group-kind', 'scry');
  // Raw offers are collapsed under "Each card's choice".
  await expect(page.getByTestId('per-creature-choices')).not.toHaveAttribute('open');
  await expect(page.getByTestId('per-creature-choices').locator('summary')).toHaveText("Each card's choice");

  const confirm = page.getByTestId('arrange-confirm');
  await expect(confirm).toBeDisabled();
  await expect(confirm).toHaveText(/Place 2 more cards/);

  // Place Opt on bottom, Island on top.
  await panel.getByRole('button', { name: 'Put Opt on bottom' }).click();
  await expect(confirm).toBeDisabled();
  await expect(confirm).toHaveText(/Place 1 more card/);

  await panel.getByRole('button', { name: 'Put Island on top' }).click();
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
  await expect(confirm).toBeEnabled();
  await expect(confirm).toHaveText(/Confirm \(1 top, 1 bottom\)/);

  // Confirm sends exactly one group_command.
  await confirm.click();
  await expect.poll(() => recorded.groupCommands.length).toBe(1);
  expect(recorded.commands).toEqual([]);
  expect(recorded.groupCommands[0].plan).toEqual({
    kind: 'scry',
    top: [rid(72)],
    bottom: [rid(71)],
  });
  await expect(panel).toHaveCount(0);
});

test('opening the rail disclosure lets a player take one raw Scry step and updates committed summary', async ({ page }) => {
  const step0 = makeFrame(
    30,
    'SCRY',
    'Scry 3',
    [
      offer(0, 'Put Opt on top', 'SCRY_TOP', [71]),
      offer(1, 'Put Island on top', 'SCRY_TOP', [72]),
      offer(2, 'Put Bolt on top', 'SCRY_TOP', [73]),
    ],
    obs => {
      obs.agent.inspected = [inspectedCard(71, 'Opt'), inspectedCard(72, 'Island'), inspectedCard(73, 'Bolt')];
    },
  );
  const group0: LiveGroup = {
    match_id: step0.match_id,
    revision: step0.revision,
    prompt_id: step0.prompt!.id,
    frame_hash: step0.frame_hash,
    kind: 'scry',
    items: [
      { item: rid(71), label: 'Opt' },
      { item: rid(72), label: 'Island' },
      { item: rid(73), label: 'Bolt' },
    ],
    committed_top: [],
    committed_bottom: [],
  };

  const step1 = makeFrame(
    31,
    'SCRY',
    'Scry (2 left)',
    [
      offer(0, 'Put Island on top', 'SCRY_TOP', [72]),
      offer(1, 'Put Bolt on top', 'SCRY_TOP', [73]),
    ],
    obs => {
      obs.agent.inspected = [inspectedCard(71, 'Opt'), inspectedCard(72, 'Island'), inspectedCard(73, 'Bolt')];
    },
  );
  const group1: LiveGroup = {
    match_id: step1.match_id,
    revision: step1.revision,
    prompt_id: step1.prompt!.id,
    frame_hash: step1.frame_hash,
    kind: 'scry',
    items: [{ item: rid(72), label: 'Island' }, { item: rid(73), label: 'Bolt' }],
    committed_top: [rid(71)],
    committed_bottom: [],
  };

  const recorded = await mountScenario(page, { frame: step0, group: group0 }, {
    onCommand: () => ({ frame: step1, group: group1 }),
  });

  // Open the collapsed disclosure and click one raw offer.
  const raw = page.getByTestId('per-creature-choices');
  await raw.locator('summary').click();
  await expect(raw).toHaveJSProperty('open', true);
  await raw.getByRole('button', { name: 'Put Opt on top' }).click();

  await expect.poll(() => recorded.commands.length).toBe(1);
  expect(recorded.commands[0].offer_id).toBe(0);
  expect(recorded.groupCommands).toEqual([]);

  // Panel now shows the committed card and only the two remaining cards.
  await expect(page.getByTestId('arrange-committed')).toContainText('Already placed');
  await expect(page.getByTestId('arrange-committed')).toContainText('Top: Opt');
  await expect(page.getByTestId('arrange-unplaced').getByTestId('arrange-card')).toHaveCount(2);
});

test('Surveil 2 transitions cleanly into OrderLibrary across two grouped Confirms', async ({ page }) => {
  const surveilFrame = makeFrame(
    40,
    'SURVEIL',
    'Surveil 2',
    [
      offer(0, 'Put Forest in graveyard', 'SELECT_CARD', [81]),
      offer(1, 'Put Esper Origins in graveyard', 'SELECT_CARD', [82]),
      offer(2, 'Keep on top', 'DECLINE_CHOICE', []),
    ],
    obs => {
      obs.agent.inspected = [inspectedCard(81, 'Forest'), inspectedCard(82, 'Esper Origins')];
    },
  );
  const surveilGroup: LiveGroup = {
    match_id: surveilFrame.match_id,
    revision: surveilFrame.revision,
    prompt_id: surveilFrame.prompt!.id,
    frame_hash: surveilFrame.frame_hash,
    kind: 'surveil',
    items: [{ item: rid(81), label: 'Forest' }, { item: rid(82), label: 'Esper Origins' }],
    committed_graveyard: [],
  };

  const orderFrame = makeFrame(
    41,
    'ORDER_LIBRARY',
    'Order top of library',
    [
      offer(0, 'Put Forest on top', 'SELECT_CARD', [81]),
      offer(1, 'Put Esper Origins on top', 'SELECT_CARD', [82]),
    ],
    obs => {
      obs.agent.inspected = [inspectedCard(81, 'Forest'), inspectedCard(82, 'Esper Origins')];
    },
  );
  const orderGroup: LiveGroup = {
    match_id: orderFrame.match_id,
    revision: orderFrame.revision,
    prompt_id: orderFrame.prompt!.id,
    frame_hash: orderFrame.frame_hash,
    kind: 'order_library',
    end: 'top',
    items: [{ item: rid(81), label: 'Forest' }, { item: rid(82), label: 'Esper Origins' }],
    committed: [],
  };

  const doneFrame = makeFrame(42, 'PRIORITY', 'Your priority', [
    { id: 0, actor: 0, verb: 'pass_priority', source: null, label: 'Pass priority', help: null, choices: [], confirm_label: 'Pass', action_type: 'PRIORITY_PASS_PRIORITY', focus: [] },
  ]);

  const recorded = await mountScenario(page, { frame: surveilFrame, group: surveilGroup }, {
    onGroupCommand: (command) => {
      if (command.plan.kind === 'surveil') return { frame: orderFrame, group: orderGroup };
      return { frame: doneFrame, group: null };
    },
  });

  // Surveil starts with both cards in Keep on top and explicit dynamic Confirm label.
  const panel = page.getByTestId('arrange-panel');
  await expect(panel).toHaveAttribute('data-group-kind', 'surveil');
  const confirm = page.getByTestId('arrange-confirm');
  await expect(confirm).toHaveText(/Keep all 2 on top/);
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');

  // Toggle one card to graveyard and back to verify dynamic label updates.
  await panel.getByRole('button', { name: 'Put Forest in graveyard' }).click();
  await expect(confirm).toHaveText(/Put 1 in graveyard, keep 1 on top/);
  await panel.getByRole('button', { name: 'Keep Forest on top' }).click();
  await expect(confirm).toHaveText(/Keep all 2 on top/);
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');

  // Confirm Surveil -> sends 1 group_command and opens OrderLibrary.
  await confirm.click();
  await expect.poll(() => recorded.groupCommands.length).toBe(1);
  expect(recorded.groupCommands[0].plan).toEqual({ kind: 'surveil', graveyard: [] });

  await expect(panel).toHaveAttribute('data-group-kind', 'order_library');
  await expect(panel.getByTestId('arrange-bucket-order')).toContainText('Top of library — drawn first at top');
  await expect(confirm).toBeDisabled();

  // Order Esper Origins first, then Forest; then reorder with keyboard/button.
  await panel.getByRole('button', { name: 'Add Esper Origins to order' }).click();
  await panel.getByRole('button', { name: 'Add Forest to order' }).click();
  await panel.getByRole('button', { name: 'Move Forest up' }).click();
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
  await confirm.click();

  await expect.poll(() => recorded.groupCommands.length).toBe(2);
  expect(recorded.groupCommands[1].plan).toEqual({
    kind: 'order_library',
    order: [rid(81), rid(82)],
  });
  await expect(panel).toHaveCount(0);
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
});
