import { readdirSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import { fitRow, offerPlacements, splitHalf, tableView, type PermanentItem, type TableView } from './view';

const FIXTURES = new URL('../../../../../managym/academy/fixtures/', import.meta.url);
const PROTOCOL = new URL('../../../../../protocol/fixtures/', import.meta.url);

/** Every retained protocol-v1 frame: anything with a projection and offers. */
function framesIn(value: unknown, found: ExperienceFrame[] = []): ExperienceFrame[] {
  if (Array.isArray(value)) value.forEach(item => framesIn(item, found));
  else if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.offers) && record.projection && 'revision' in record) found.push(record as unknown as ExperienceFrame);
    Object.values(record).forEach(item => framesIn(item, found));
  }
  return found;
}

function load(url: URL): unknown {
  return JSON.parse(readFileSync(url, 'utf8'));
}

const retained = [FIXTURES, PROTOCOL].flatMap(dir => readdirSync(dir).filter(name => name.endsWith('.json'))
  .flatMap(name => framesIn(load(new URL(name, dir))).map(frame => ({ name, frame }))));

const navigation = (load(new URL('choice-navigation-w17-smart-payment.json', FIXTURES)) as {
  positions: Record<string, { frame: ExperienceFrame }>;
}).positions;

function permanents(view: TableView): PermanentItem[] {
  return [view.you, view.opponent].flatMap(side => [...side.creatures, ...side.support])
    .flatMap(pile => pile.items).flatMap(item => [item, ...item.attached]);
}

describe('tableView over every retained frame', () => {
  it('finds retained frames to check', () => {
    expect(retained.length).toBeGreaterThan(10);
  });

  it.each(retained.map(({ name, frame }) => [`${name} r${frame.revision} ${frame.action_space}`, frame] as const))(
    '%s: every offer is listed once and drawn at least once',
    (_, frame) => {
      const view = tableView(frame);
      expect(view.list.map(placed => placed.offer.id)).toEqual(frame.offers.map(offer => offer.id));
      expect(view.list.map(placed => placed.number)).toEqual(frame.offers.map((_, index) => index + 1));
      const placements = offerPlacements(view);
      for (const offer of frame.offers) expect(placements.get(offer.id) ?? 0, offer.label).toBeGreaterThanOrEqual(1);
    },
  );

  it.each(retained.map(({ name, frame }) => [`${name} r${frame.revision}`, frame] as const))(
    '%s: every permanent is drawn exactly once',
    (_, frame) => {
      const drawn = permanents(tableView(frame)).map(item => item.permanent.id).sort((a, b) => a - b);
      const expected = [...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield]
        .map(permanent => permanent.id).sort((a, b) => a - b);
      expect(drawn).toEqual(expected);
    },
  );
});

describe('where decisions land', () => {
  it('puts targets on the permanents they target', () => {
    const view = tableView(navigation.CHOOSE_TARGET.frame);
    const targeted = permanents(view).filter(item => item.offers.length);
    expect(targeted.map(item => item.permanent.id).sort()).toEqual(
      navigation.CHOOSE_TARGET.frame.offers.map(offer => offer.focus[0]).sort());
    expect(view.tray).toEqual([]);
  });

  it('puts payment taps on lands and the payment on the primary button', () => {
    const view = tableView(navigation.PAY_OR_NOT.frame);
    const lands = view.you.support.flatMap(pile => pile.items).filter(item => item.offers.length);
    expect(lands.flatMap(item => item.offers.map(placed => placed.offer.label)))
      .toEqual(expect.arrayContaining(['Tap Mountain to help pay', 'Tap Island to help pay']));
    expect(view.primary?.offer.label).toBe('Pay the cost');
  });

  it('draws Learn: discards on hand cards, Lessons in the tray by name, declining as a button', () => {
    const view = tableView(navigation.LEARN.frame);
    expect(view.hand.filter(item => item.offers.length).length).toBe(6);
    expect(view.tray.map(item => item.name)).toEqual(['Accumulate Wisdom', 'Firebending Lesson', "It'll Quench Ya!"]);
    expect(view.buttons.map(placed => placed.offer.label)).toEqual(['Decline Learn']);
    // A click on the Learn spell on the stack never declines.
    expect(view.stack.flatMap(item => item.offers)).toEqual([]);
  });

  it('asks about one blocker: blocks on attackers, not blocking as a button', () => {
    const view = tableView(navigation.DECLARE_BLOCKER.frame);
    const blocker = navigation.DECLARE_BLOCKER.frame.offers[0].focus[0];
    expect(view.asked).toEqual(new Set([blocker]));
    expect(view.buttons.map(placed => placed.offer.label)).toEqual(['Otter-Penguin: do not block']);
    const attackers = view.opponent.creatures.flatMap(pile => pile.items).filter(item => item.offers.length);
    expect(attackers.length).toBe(1);
    expect(view.primary).toBeNull();
  });

  it('attacks by clicking the creature asked about; declining stays a button', () => {
    const canonical = retained.find(({ frame }) => frame.action_space === 'DECLARE_ATTACKER')!.frame;
    const frame = structuredClone(canonical);
    const creature = frame.offers[0].focus[0];
    const subject = { entity_id: 1, subject: { kind: 'object' as const, id: { entity: creature, incarnation: 0 } }, display_id: creature, label: 'Flyer' };
    // As the authority names them: the attacker is the declaration's source; its decline names it as a choice.
    frame.offers[0].subjects = [{ role: 'source', object: subject }];
    frame.offers[1].subjects = [{ role: 'choice', object: subject }];
    const view = tableView(frame);
    const attacker = permanents(view).find(item => item.permanent.id === creature)!;
    expect(attacker.offers.map(placed => placed.offer.label)).toEqual([frame.offers[0].label]);
    expect(view.buttons.map(placed => placed.offer.label)).toEqual([frame.offers[1].label]);
    expect(view.asked.has(creature)).toBe(true);
  });

  it('fans identical lands into one pile, keeping each copy', () => {
    const view = tableView(navigation.CHOOSE_TARGET.frame);
    const piles = [...view.you.support, ...view.you.creatures];
    expect(piles.some(pile => pile.items.length > 1)).toBe(true);
    expect(piles.flatMap(pile => pile.items).length).toBe(navigation.CHOOSE_TARGET.frame.projection.agent.battlefield.length);
  });
});

describe('fitRow', () => {
  const limits = { gap: 8, minCard: 44, maxCard: 140, maxLines: 3, ratio: 5 / 7 };
  const upright = (n: number) => Array.from({ length: n }, () => 1);

  it('uses the full height for a small board', () => {
    const fit = fitRow(upright(3), 800, 150, limits);
    expect(fit.lines).toBe(1);
    expect(fit.card).toBe(Math.floor(150 * 5 / 7));
  });

  it('sizes landscape frames by their own ratio', () => {
    const fit = fitRow(upright(3), 800, 90, { ...limits, maxCard: 200, ratio: 4 / 3 });
    expect(fit.card).toBe(120);
  });

  it('keeps twenty-four distinct creatures readable by wrapping', () => {
    const fit = fitRow(upright(24), 820, 300, limits);
    expect(fit.overflow).toBe(false);
    expect(fit.card).toBeGreaterThanOrEqual(55);
    expect(fit.lines).toBeGreaterThan(1);
  });

  it('fits a pile of twelve lands in less width than twelve cards', () => {
    const pile = fitRow([12], 400, 120, limits);
    const spread = fitRow(upright(12), 400, 120, limits);
    expect(pile.card).toBeGreaterThan(spread.card);
  });

  it('reports overflow only past the smallest size', () => {
    expect(fitRow(upright(80), 600, 100, limits).overflow).toBe(true);
  });
});

describe('splitHalf', () => {
  const limits = { gap: 6, creatures: { gap: 8, minCard: 40, maxCard: 132, maxLines: 3, ratio: 5 / 7 }, support: { gap: 8, minCard: 36, maxCard: 104, maxLines: 2, ratio: 5 / 7 } };
  const upright = (n: number) => Array.from({ length: n }, () => 1);

  it('gives a crowded creature row the height a few land piles do not need', () => {
    const split = splitHalf(upright(18), [5, 5], 840, 280, limits);
    expect(split.creatures).toBeGreaterThan(280 * 0.58);
    expect(split.creatureFit.card).toBeGreaterThanOrEqual(60);
    expect(split.creatureFit.overflow || split.supportFit.overflow).toBe(false);
  });

  it('keeps lands readable when there are no creatures', () => {
    const split = splitHalf([], upright(8), 840, 280, limits);
    expect(split.supportFit.card).toBeGreaterThan(80);
  });
});
