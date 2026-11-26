import { expect, test, type Page } from '@playwright/test';
import bolt from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { ExperienceFrame, PermanentState, RecoveryEnvelope } from '../src/lib/types';
import { denseFrame } from './fixtures/laptop-board';
import { gameReply } from './fixtures/game';

/** Setup with the matchup named, so first-visit defaults do not choose one. */
const SETUP = '/?hero=ur_lessons&villain=gw_allies';

function badgeFrame(): ExperienceFrame {
  const frame = denseFrame();
  let id = 700;
  function creature(name: string, power: number, toughness: number, state: Partial<PermanentState> = {}): PermanentState {
    return { id: ++id, name, power, toughness, base_power: power, base_toughness: toughness, controller_id: 0,
      tapped: false, summoning_sick: false, damage: 0, plus1_counters: 0, ...state };
  }
  const bear = { keywords: ['defender'], keyword_grants: {} };
  const prowess = [{ label: 'Prowess' }];
  const { agent, opponent } = frame.projection;
  agent.battlefield = [
    creature('Platypus-Bear', 2, 3, { ...bear, attacks_despite_defender: true }),
    creature('Mai, Jaded Edge', 1, 3, { summoning_sick: true, keyword_abilities: prowess }),
    creature('Fire Nation Cadets', 1, 2, { keyword_abilities: [{ label: 'Firebending 2' }] }),
    creature('Sokka, Tenacious Tactician', 3, 3),
    creature('Ally', 1, 1, { is_token: true, keywords: ['menace'], keyword_grants: { menace: { source: 'Sokka, Tenacious Tactician' } },
      keyword_abilities: [{ label: 'Prowess', granted: true, source: 'Sokka, Tenacious Tactician' }] }),
  ];
  opponent.battlefield = [
    creature('Platypus-Bear', 2, 3, bear),
    creature('Ty Lee, Chi Blocker', 2, 1, { keywords: ['flash'], keyword_grants: {}, keyword_abilities: prowess }),
    creature('Fire Nation Cadets', 1, 2, { restrictions: [{ kind: 'doesnt_untap', source: 'Watery Grasp' }] }),
    creature('Soldier', 1, 1, { is_token: true }),
    creature('Soldier', 1, 1, { is_token: true, keyword_abilities: [{ label: 'Firebending 1' }] }),
    creature('Sire of Seven Deaths', 7, 7, { keywords: ['first_strike', 'vigilance', 'menace', 'trample', 'reach', 'lifelink'],
      keyword_grants: {}, keyword_abilities: [{ label: 'Ward—Pay 7 life' }, { label: 'Ward—Discard a card', granted: true, effect_timestamp: 9 }],
      effects: [{ modification: 'GrantStatic', timestamp: 9, source: 'Shield', duration: 'EndOfTurn' }] }),
  ].map(card => ({ ...card, controller_id: 1 }));
  agent.hand = agent.hand.slice(0, 3);
  agent.zone_counts.HAND = agent.hand.length;
  return frame;
}

async function show(page: Page, frame: ExperienceFrame) {
  const game = gameReply();
  let publish = () => {};
  await page.routeWebSocket('**/ws/play', socket => {
    publish = () => socket.send(JSON.stringify({
      type: 'observation', data: frame.projection, actions: [], ...game,
      recovery: { ...bolt.recovery as RecoveryEnvelope, frame, accepted_commands: [], presentation_tail: [], replay_cursor: frame.revision },
    }));
    socket.onMessage(raw => {
      if (['new_game', 'resume'].includes(JSON.parse(String(raw)).type)) publish();
    });
  });
  await page.goto(SETUP);
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  await expect(page.getByTestId('game')).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  return () => {
    frame.revision++;
    frame.frame_hash = `badges-${frame.revision}`;
    publish();
  };
}

async function wholeChips(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  const { count, collisions } = await page.evaluate(() => {
    const chips = [...document.querySelectorAll<HTMLElement>('button.field .chip')];
    const collisions = chips.flatMap(chip => {
      const card = chip.closest<HTMLElement>('button.field')!;
      const rect = chip.getBoundingClientRect();
      const art = chip.closest('.art')?.getBoundingClientRect();
      const overlaps = [...card.querySelectorAll('.pt, .policy')].some(plate => {
        const p = plate.getBoundingClientRect();
        return rect.left < p.right && p.left < rect.right && rect.top < p.bottom && p.top < rect.bottom;
      });
      return chip.scrollWidth > chip.clientWidth || overlaps || (art && (
        rect.left < art.left || rect.right > art.right || rect.top < art.top || rect.bottom > art.bottom
      )) ? [`${card.dataset.cardName} ${chip.textContent}`] : [];
    });
    return { count: chips.length, collisions };
  });
  expect(count).toBeGreaterThan(0);
  expect(collisions).toEqual([]);
}

for (const colorScheme of ['light', 'dark'] as const) {
  test(`battlefield badges and inspection in ${colorScheme} at 1280×800`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
    const frame = badgeFrame();
    // In dark mode the card-face service is unavailable. The same effective
    // badge, provenance and inspection assertions run in both themes.
    if (colorScheme === 'dark') await page.route('**/api/card-faces?**', route => route.fulfill({ json: {} }));
    const publish = await show(page, frame);
    const card = (id: number) => page.locator(`button.field[data-object-id="${id}"]`);
    const chips = (id: number) => card(id).locator('.chip');
    await expect(chips(701)).toHaveText(['Defender']);
    await expect(chips(701)).toHaveCSS('text-decoration-line', 'line-through');
    await expect(chips(706)).toHaveCSS('text-decoration-line', 'none');
    await expect(card(702).locator('.sick svg')).toHaveAttribute('width', '16');
    await expect(card(702).locator('.sick')).toHaveAttribute('title', 'Summoning sick');
    await expect(chips(702).last()).toHaveText('Prowess');
    await expect(chips(703)).toHaveText(['Firebending 2']);
    await expect(card(705).locator('.chip.granted')).toHaveText(['Menace', 'Prowess']);
    await expect(chips(707)).toHaveText(['Prowess']);
    await expect(card(707)).toHaveAccessibleName(/Flash/);
    await expect(chips(708)).toHaveText(["Doesn't untap"]);
    await expect(chips(709)).toHaveCount(0);
    await expect(chips(710)).toHaveText(['Firebending 1']);
    await expect(page.getByTestId('pile-count-badge')).toHaveCount(0);
    await expect(chips(711).last()).toHaveText(/^\+\d+$/);
    await wholeChips(page);
    await page.screenshot({ path: testInfo.outputPath('battlefield.png') });

    const inspect = page.getByTestId('card-preview');
    await card(701).focus();
    await expect(inspect).toContainText('Defender · can attack anyway');
    await card(705).focus();
    await expect(inspect).toContainText('Prowess · Sokka, Tenacious Tactician');
    await card(707).focus();
    await expect(inspect).toContainText('Flash');
    await card(711).focus();
    await expect(inspect).toContainText('Ward—Pay 7 life');
    await expect(inspect).toContainText('Ward—Discard a card · Shield, until end of turn');

    await page.screenshot({ path: testInfo.outputPath('inspection.png') });

    // A later grant survives ability loss. Removing that grant on the next
    // frame must clear it even though the card keeps its public id and face.
    const stripped = frame.projection.agent.battlefield[1];
    stripped.abilities_lost = true;
    stripped.keyword_abilities = [{ label: 'Ward {2}', granted: true, effect_timestamp: 11 }];
    stripped.effects = [
      { modification: 'LoseAllAbilities', timestamp: 10, source: 'Frogify', duration: 'Indefinite' },
      { modification: 'GrantStatic', timestamp: 11, source: 'Shield', duration: 'EndOfTurn' },
    ];
    publish();
    await expect(card(702)).toHaveAccessibleName(/Lost base abilities · Frogify.*Ward \{2\} · Shield/);
    await expect(card(702).locator('.chip.granted')).toHaveText('Ward {2}');
    delete stripped.keyword_abilities;
    stripped.effects.pop();
    publish();
    await expect(card(702)).toHaveAccessibleName(/No abilities · Frogify/);
    await expect(card(702).locator('.chip.granted')).toHaveCount(0);

    // Crowd the same row. The copy's hidden facts remain available after the
    // resize, including if a two-line name and a turned card leave no chip row.
    const crowded = frame.projection.agent.battlefield;
    const fresh = crowded[1];
    delete fresh.abilities_lost;
    delete fresh.effects;
    fresh.keyword_abilities = [{ label: 'Prowess' }];
    fresh.keywords = ['flying', 'haste', 'trample'];
    fresh.counters = { charge: 2 };
    fresh.restrictions = [{ kind: 'cant_block' }];
    for (let i = 0; i < 18; ++i) crowded.push({ ...fresh, id: 800 + i, tapped: i % 2 === 0, plus1_counters: i + 1 });
    publish();
    await expect(card(817)).toBeVisible();
    await expect(card(802).locator('.overflow')).toHaveText(/^\+\d+$/);
    await expect(card(802).locator('.overflow')).toHaveAttribute('title', /Prowess|Trample/);
    await expect(card(802)).toHaveAccessibleName(/Summoning sick.*Can't block.*Prowess/);
    expect((await card(817).boundingBox())!.width).toBeLessThan(80);
    await wholeChips(page);
    await page.mouse.move(4, 4);
    await page.screenshot({ path: testInfo.outputPath('crowded.png') });
  });
}

for (const colorScheme of ['light', 'dark'] as const) {
  test(`headline abilities fit beside lands and a four-card hand in ${colorScheme}`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
    const frame = denseFrame();
    const { agent, opponent } = frame.projection;
    agent.hand = agent.hand.slice(0, 4);
    agent.zone_counts.HAND = 4;
    const [cadets, tyLee, unagi] = agent.battlefield.filter(p => p.power !== null).slice(0, 3);
    Object.assign(cadets, { name: 'Fire Nation Cadets', power: 1, toughness: 2, tapped: false, damage: 0, keyword_abilities: [{ label: 'Firebending 2' }] });
    Object.assign(tyLee, { name: 'Ty Lee, Chi Blocker', power: 2, toughness: 1, keywords: ['flash'], tapped: true, restrictions: [{ kind: 'doesnt_untap', source: 'Watery Grasp' }],
      keyword_abilities: [{ label: 'Prowess' }] });
    Object.assign(unagi, { name: 'The Unagi of Kyoshi Island', power: 5, toughness: 5, keywords: ['flash'], tapped: false, keyword_abilities: [{ label: 'Ward—Waterbend {4}' }] });
    agent.battlefield = [cadets, tyLee, unagi, ...agent.battlefield.filter(p => p.types?.is_land)];
    opponent.battlefield = opponent.battlefield.filter(p => p.types?.is_land);
    await show(page, frame);
    const card = (id: number) => page.locator(`button.field[data-object-id="${id}"]`);
    await expect(card(cadets.id).locator('.chip')).toHaveText([/^Firebend(ing)? 2$/]);
    await expect(card(tyLee.id).locator('.chip')).toHaveText(["Doesn't untap", 'Prowess']);
    await expect(card(unagi.id).locator('.chip')).toHaveText(['Ward']);
    await expect(card(unagi.id).locator('.chip')).toHaveAttribute('title', 'Ward—Waterbend {4}');
    // The board chooses the card size from both rows and the hand.
    expect((await card(cadets.id).boundingBox())!.width).toBeLessThanOrEqual(82);
    await wholeChips(page);
    await page.setViewportSize({ width: 1280, height: 788 });
    await expect.poll(async () => (await card(cadets.id).boundingBox())!.width).toBeLessThanOrEqual(80);
    await expect(card(cadets.id).locator('.chip')).toHaveText([/^Firebend(ing)? 2$/]);
    await expect(card(tyLee.id).locator('.chip')).toHaveText(["Doesn't untap", 'Prowess']);
    await wholeChips(page);
  });
}

test('recorded Watch badges leave the policy percentage visible', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const frame = badgeFrame();
  const mai = frame.projection.agent.battlefield[1];
  frame.offers = [{ ...frame.offers[0], verb: 'activate', action_type: 'PRIORITY_ACTIVATE_ABILITY',
    label: 'Activate Mai', focus: [mai.id] }];
  const choice = { key: 'mai', kind: 'choice', turn: 9, phase: 'PRECOMBAT_MAIN', step: 'PRECOMBAT_MAIN_STEP',
    actor: 0, automatic: false, can_reveal: false, revision: frame.revision };
  await page.route('**/api/traces/badges/recording', route => route.fulfill({ json: {
    id: 'badges', positions: [choice], total: 1, next_offset: null, revision: frame.revision, status: 'finished',
    can_produce: false, can_study: false, winner: null, players: [], source_kind: null, watch: true,
  } }));
  await page.route('**/api/traces/badges/positions/**', route => route.fulfill({ json: {
    ...choice, observation: frame.projection, decision: { frame, viewer: 0, offer_id: frame.offers[0].id },
    outputs: { probabilities: [1], search_scores: [] }, viewer: 0, revealed: false, description: 'Activate Mai',
    note_key: 'mai', history: [], history_labels: [], history_before: null, cursor: 0, transition_start: 0, consequences_recorded: true,
  } }));
  await page.goto('/replay?trace=badges&position=mai');
  await page.evaluate(() => document.fonts.ready);
  const card = page.locator(`button.field[data-object-id="${mai.id}"]`);
  await expect(card.getByTestId('card-policy')).toContainText('100%');
  await expect(card.locator('.chip.ability')).toHaveText('Prowess');
  const policy = (await card.getByTestId('card-policy').boundingBox())!;
  const chips = (await card.locator('.chips').boundingBox())!;
  expect(chips.y + chips.height).toBeLessThan(policy.y);
  await wholeChips(page);
});
