import { expect, test, type Page } from '@playwright/test';
import bolt from '../../../protocol/fixtures/bolt-target.json' with { type: 'json' };
import type { ExperienceFrame, PermanentState, RecoveryEnvelope } from '../src/lib/types';
import { denseFrame } from './fixtures/laptop-board';
import { tableSnapshot } from './fixtures/table';

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
    creature('Ally', 1, 1, { is_token: true, keywords: ['menace'], keyword_grants: { menace: null },
      keyword_abilities: [{ label: 'Prowess', granted: true }] }),
  ];
  opponent.battlefield = [
    creature('Platypus-Bear', 2, 3, bear),
    creature('Ty Lee, Chi Blocker', 2, 1, { keywords: ['flash'], keyword_grants: {}, keyword_abilities: prowess }),
    creature('Fire Nation Cadets', 1, 2, { restrictions: ['doesnt_untap'] }),
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
  const table = tableSnapshot();
  let publish = () => {};
  await page.routeWebSocket('**/ws/play', socket => {
    publish = () => socket.send(JSON.stringify({
      type: 'observation', data: frame.projection, actions: [], table, session_id: 'badges', resume_token: 'badges',
      recovery: { ...bolt.recovery as RecoveryEnvelope, frame, accepted_commands: [], presentation_tail: [], replay_cursor: frame.revision },
    }));
    socket.onMessage(raw => {
      if (['new_game', 'resume'].includes(JSON.parse(String(raw)).type)) publish();
    });
  });
  await page.goto('/?setup=1');
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
  const chips = page.locator('.field .chip');
  expect(await chips.count()).toBeGreaterThan(0);
  // offset coordinates are local, before a card's combat rotation.
  expect(await chips.evaluateAll(elements => elements.flatMap(chip => {
    const el = chip as HTMLElement;
    const box = el.closest<HTMLElement>('.chips');
    if (el.scrollWidth > el.clientWidth || (box && (el.offsetLeft + el.offsetWidth > box.clientWidth ||
      box.offsetTop < 0 || box.offsetTop + box.offsetHeight > box.parentElement!.clientHeight))) {
      return [el.textContent];
    }
    return [];
  }))).toEqual([]);
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
    fresh.restrictions = ['cant_block'];
    for (let i = 0; i < 18; ++i) crowded.push({ ...fresh, id: 800 + i, tapped: i % 2 === 0, plus1_counters: i + 1 });
    publish();
    await expect(card(817)).toBeVisible();
    await expect(card(802).locator('.overflow')).toHaveText(/^\+\d+$/);
    await expect(card(802)).toHaveAccessibleName(/Summoning sick.*Can't block.*Prowess/);
    expect((await card(817).boundingBox())!.width).toBeLessThan(80);
    await wholeChips(page);
    await page.mouse.move(4, 4);
    await page.screenshot({ path: testInfo.outputPath('crowded.png') });
  });
}
