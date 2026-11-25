import { expect, test, type Page } from '@playwright/test';
import { combatObservation, startCombat } from './fixtures/combat';

const card = (page: Page, id: number) => page.locator(`.board [data-testid=board-card][data-object-id="${id}"]`);

async function press(page: Page, mode: 'pointer' | 'keyboard', id: number): Promise<void> {
  if (mode === 'pointer') await card(page, id).click();
  else { await card(page, id).focus(); await page.keyboard.press('Enter'); }
}

for (const mode of ['pointer', 'keyboard'] as const) {
  test(`combat is drafted on the board by ${mode} and submitted in one group_command on confirm`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const authority = await startCombat(page);

    // Every creature that can attack is outlined, and the primary reads the whole plan.
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'can');
    await expect(card(page, 85)).toHaveAttribute('data-draft', 'can');
    await expect(card(page, 83)).toHaveAccessibleName(/can attack; press to add it to the attack/);
    await expect(page.getByTestId('primary-action')).toHaveText(/No attack/);
    // The draft is the declaration: each creature's own offer waits, closed, under the legal choices, and is not a numbered key.
    await expect(page.getByTestId('per-creature-choices')).not.toHaveAttribute('open');
    await expect(page.getByTestId('action-option').first()).toBeHidden();
    await page.keyboard.press('1');

    // Drafting sends nothing: Otter-Penguin leans in, Tiger-Seal stays home.
    await press(page, mode, 83);
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'in');
    await expect(card(page, 83)).toHaveClass(/attacking/);
    await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with Otter-Penguin · not with Tiger-Seal');
    await press(page, mode, 85);
    await press(page, mode, 85);
    // The button fits one line; the prompt above names the plan.
    await expect(page.getByTestId('primary-action')).toHaveText(/Attack with 1/);
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.commands).toEqual([]);
    expect(authority.groupCommands).toEqual([]);

    // Confirm sends exactly one atomic group_command.
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.groupCommands.length).toBe(1);
    expect(authority.commands).toEqual([]);
    expect(authority.groupCommands[0]).toMatchObject({
      expected_revision: 10,
      prompt_id: 110,
      frame_hash: 'combat-frame-10',
      plan: { kind: 'declare_attackers', attackers: [{ entity: 83, incarnation: 0 }] },
    });

    // Blocking: Otter-Penguin can block only the Cub, so pressing it drafts that block and draws its line.
    await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
    await expect(page.getByTestId('combat-draft')).toHaveAttribute('data-kind', 'blockers');
    await expect(card(page, 83)).toHaveAccessibleName(/can block; press, then the attacker it blocks/);
    await press(page, mode, 83);
    await expect(card(page, 83)).toHaveAttribute('data-draft', 'in');
    await expect(page.locator('[data-testid=board-lines] [data-line-kind=block]')).toHaveCount(1);
    await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin on Badgermole Cub');
    await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
    expect(authority.groupCommands).toHaveLength(1);
    if (mode === 'pointer') await page.getByTestId('primary-action').click();
    else await page.keyboard.press('Space');
    await expect.poll(() => authority.groupCommands.length).toBe(2);
    expect(authority.groupCommands[1]).toMatchObject({
      expected_revision: 12,
      prompt_id: 112,
      frame_hash: 'combat-frame-12',
      plan: {
        kind: 'declare_blockers',
        blocks: [{ blocker: { entity: 83, incarnation: 0 }, attacker: { entity: 84, incarnation: 0 } }],
      },
    });
    expect(errors).toEqual([]);
  });
}

test('a confirmed attack stays drafted and disabled while the group_command is pending', async ({ page }) => {
  const work = Promise.withResolvers<void>();
  const authority = await startCombat(page, { finishWork: work.promise });
  await card(page, 85).click();
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.getByTestId('primary-action').click();
  await expect.poll(() => authority.groupCommands.length).toBe(1);
  // While pending, the draft stays drawn and disabled with no Stop sending button.
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'false');
  await expect(page.getByTestId('primary-action')).toBeDisabled();
  await expect(page.getByTestId('combat-stop')).toHaveCount(0);
  await expect(page.getByTestId('per-creature-choices').locator('summary')).toBeVisible();
  await expect(page.getByTestId('action-option').first()).toBeHidden();
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await expect(page.locator('.rows .asked')).toHaveCount(0);
  work.resolve();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
});

test('a group_refused rejection keeps the draft for editing with the engine reason shown', async ({ page }) => {
  const authority = await startCombat(page, {
    refuseGroupOnce: {
      item: { entity: 83, incarnation: 0 },
      label: 'Otter-Penguin',
      reason: 'Otter-Penguin must attack if able.',
    },
  });
  await card(page, 85).click();
  const confirm = page.getByTestId('primary-action');
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
  await confirm.click();
  await expect.poll(() => authority.groupCommands.length).toBe(1);

  // Frame unchanged, draft kept, engine reason shown on the prompt and refused creature.
  await expect(page.locator('main')).toHaveAttribute('data-frame-revision', '10');
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await expect(page.getByTestId('decision-prompt')).toHaveText('Otter-Penguin must attack if able.');
  await expect(card(page, 83)).toHaveAccessibleName(/Otter-Penguin must attack if able\./);
  await expect(confirm).toBeDisabled();

  // Editing the draft clears the refusal and lets the player confirm.
  await card(page, 83).click();
  await expect(confirm).toHaveAttribute('data-draft-ready', 'true');
  await expect(confirm).toBeEnabled();
  await confirm.click();
  await expect.poll(() => authority.groupCommands.length).toBe(2);
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
});

test('a pile of identical creatures takes a count', async ({ page }) => {
  const authority = await startCombat(page, { moreTigers: 2 });
  const pile = page.getByTestId('pile-count');
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 0 of 3 attacking');
  await pile.getByRole('button', { name: 'One more Tiger-Seal' }).click();
  await pile.getByRole('button', { name: 'One more Tiger-Seal' }).click();
  await expect(pile).toHaveAccessibleName('Tiger-Seal: 2 of 3 attacking');
  await expect(page.locator('.board [data-testid=board-card][data-draft=in]')).toHaveCount(2);
  await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with 2 · not with Otter-Penguin and Tiger-Seal');
  expect(authority.groupCommands).toEqual([]);
});

test('a reload mid-draft keeps the draft and commits nothing', async ({ page }) => {
  const authority = await startCombat(page);
  await card(page, 85).click();
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await page.reload();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_ATTACKER');
  await expect(card(page, 85)).toHaveAttribute('data-draft', 'in');
  await expect(card(page, 83)).toHaveAttribute('data-draft', 'can');
  await expect(page.getByTestId('decision-prompt')).toHaveText('Attack with Tiger-Seal · not with Otter-Penguin');
  expect(authority.groupCommands).toEqual([]);
  // The restored draft is sent in one group_command.
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.keyboard.press('Space');
  await expect.poll(() => authority.groupCommands.length).toBe(1);
  expect(authority.groupCommands[0].plan).toEqual({
    kind: 'declare_attackers',
    attackers: [{ entity: 85, incarnation: 0 }],
  });
});

test('after combat, the block stays drawn and damage shows on the cards it landed on', async ({ page }, testInfo) => {
  const authority = await startCombat(page);
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'DECLARE_BLOCKER');
  await card(page, 83).click();
  await expect(page.getByTestId('primary-action')).toHaveAttribute('data-draft-ready', 'true');
  await page.getByTestId('primary-action').click();
  await expect(page.getByTestId('game')).toHaveAttribute('data-action-space', 'PRIORITY');
  // No frame holds the block any longer; the block events since the move draw it.
  await expect(page.locator('[data-testid=board-lines] [data-line-kind=block]')).toHaveCount(1);
  await expect(card(page, 84).getByTestId('damage-dealt')).toHaveText('−2');
  await expect(card(page, 83).getByTestId('damage-dealt')).toHaveText('−1');
  await expect(card(page, 84)).toHaveAccessibleName(/dealt 2 since your last move/);
  await expect(page.getByTestId('recent-fact')).toHaveText([
    'Otter-Penguin blocked Badgermole Cub',
    'Badgermole Cub dealt 1 to Otter-Penguin',
    'Otter-Penguin dealt 2 to Badgermole Cub',
  ]);
  await page.screenshot({ path: testInfo.outputPath('after-combat.png') });
  // Passing is the person's next move: the marks of the last one go.
  await expect(page.getByTestId('game')).toHaveAttribute('data-live', 'true');
  const sent = authority.commands.length;
  await page.keyboard.press('Space');
  await expect.poll(() => authority.commands.length).toBe(sent + 1);
});

test('Replay keeps playing through a live producer save and resumes at its saved choice', async ({ page }) => {
  const observation = combatObservation();
  const latest = {key: 'latest', kind: 'state', turn: 4, phase: 'COMBAT', step: 'DAMAGE', actor: null, automatic: false, can_reveal: false};
  const choice = {...latest, key: 'saved-choice', kind: 'choice'};
  const consequences = {...latest, key: 'saved-consequences', kind: 'consequences'};
  let produced = 0;
  let committed = false;
  let release: (() => void) | null = null;
  await page.route('**/api/traces', route => route.fulfill({json: []}));
  await page.route('**/api/watch/pending/step', async route => {
    produced++;
    await new Promise<void>(resolve => release = resolve);
    committed = true;
    await route.fulfill({json: {}});
  });
  await page.route('**/api/traces/pending/recording', route => route.fulfill({json: {
    id: 'pending', positions: committed ? [choice, consequences, latest] : [latest],
    total: committed ? 3 : 1, next_offset: null, revision: committed ? 12 : 11,
    status: 'waiting', can_produce: true, winner: null, players: [], source_kind: null, watch: true, can_feedback: false,
  }}));
  await page.route('**/api/traces/pending/positions/**', route => {
    const key = decodeURIComponent(new URL(route.request().url()).pathname.split('/positions/')[1]);
    const position = [choice, consequences, latest].find(position => position.key === key)!;
    return route.fulfill({json: {...position, observation, decision: null, outputs: null,
      description: position.kind === 'choice' ? 'Saved producer choice' : null, note_key: key, viewer: 0, revealed: false, history: [], history_labels: [], history_before: null,
      cursor: 0, transition_start: 0, consequences_recorded: true,
    }});
  });
  await page.goto('/replay?trace=pending');
  await expect(page.getByTestId('recording-status')).toHaveText('Waiting at the live edge');
  await page.clock.install({time: 0});
  await page.clock.pauseAt(1000);
  await page.getByRole('combobox', {name: 'Speed', exact: true}).selectOption('16');
  await page.getByRole('button', {name: 'Play', exact: true}).click();
  await page.clock.runFor(100);
  await expect.poll(() => release !== null).toBe(true);
  await expect(page.getByRole('button', {name: 'Pause', exact: true})).toBeEnabled();
  release!();
  const counter = page.getByTestId('position-counter');
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByTestId('recording-status')).toHaveText('Live Watch · reading saved recording');
  await expect.poll(() => new URL(page.url()).searchParams.get('position')).toBe('saved-choice');
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await page.clock.runFor(500);
  expect(produced).toBe(1);
  await expect(counter).toHaveText('Position 1 / 3');
  await expect(page.getByTestId('game-log').locator('[aria-current="true"]')).toHaveCount(1);
  await page.reload();
  await expect(counter).toHaveText('Position 1 / 3');
  expect(produced).toBe(1);
});
