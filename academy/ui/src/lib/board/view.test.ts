import { readdirSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import { fitRow, footprint, handStrip, pileWidth, offerPlacements, possessive, splitHalf, stackProvenance, boardView, unpilePlanned, type PermanentItem, type PileShape, type BoardView } from './view';
import { shortName } from '$lib/opponent-name';
import { counterTerms, permissionTerms } from './inspect';

const PLAYERS = { you: 'You', opponent: 'Heuristic' };

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

function permanents(view: BoardView): PermanentItem[] {
  return [view.you, view.opponent].flatMap(side => [...side.creatures, ...side.support])
    .flatMap(pile => pile.items).flatMap(item => [item, ...item.attached]);
}

describe('boardView over every retained frame', () => {
  it('finds retained frames to check', () => {
    expect(retained.length).toBeGreaterThan(10);
  });

  it.each(retained.map(({ name, frame }) => [`${name} r${frame.revision} ${frame.action_space}`, frame] as const))(
    '%s: every offer is listed once and drawn at least once',
    (_, frame) => {
      const view = boardView(frame, PLAYERS);
      expect(view.list.map(placed => placed.offer.id)).toEqual(frame.offers.map(offer => offer.id));
      expect(view.list.map(placed => placed.number)).toEqual(frame.offers.map((_, index) => index + 1));
      const placements = offerPlacements(view);
      for (const offer of frame.offers) expect(placements.get(offer.id) ?? 0, offer.label).toBeGreaterThanOrEqual(1);
    },
  );

  it.each(retained.map(({ name, frame }) => [`${name} r${frame.revision}`, frame] as const))(
    '%s: every permanent is drawn exactly once',
    (_, frame) => {
      const drawn = permanents(boardView(frame, PLAYERS)).map(item => item.permanent.id).sort((a, b) => a - b);
      const expected = [...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield]
        .map(permanent => permanent.id).sort((a, b) => a - b);
      expect(drawn).toEqual(expected);
    },
  );
});

describe('what is committed to the pending cost', () => {
  const pay = retained.map(({ frame }) => frame)
    .find(frame => frame.projection.decision_context && frame.projection.agent.battlefield.length >= 2)!;
  const [ogre, other] = pay.projection.agent.battlefield;
  const paying: ExperienceFrame = { ...pay, projection: { ...pay.projection, decision_context: { ...pay.projection.decision_context!, paid: [
    { phrase: `sacrificed ${ogre.name}`, object: { entity_id: 0, subject: { kind: 'object', id: { entity: ogre.id, incarnation: 0 } }, display_id: ogre.id, label: ogre.name ?? '' } },
    { phrase: 'exiled Opt to delve' },
  ] } } };

  it('marks the permanent that paid with the engine\'s phrase, alone in its pile', () => {
    const view = boardView(paying, PLAYERS);
    const marked = permanents(view).filter(item => item.paid !== null);
    expect(marked.map(item => [item.permanent.id, item.paid])).toEqual([[ogre.id, `sacrificed ${ogre.name}`]]);
    const pile = [...view.you.creatures, ...view.you.support].find(candidate => candidate.items.some(item => item.permanent.id === ogre.id))!;
    expect(pile.items.map(item => item.permanent.id)).toEqual([ogre.id]);
    expect(permanents(view).find(item => item.permanent.id === other.id)?.paid).toBeNull();
  });

  it('marks nothing when nothing has paid', () => {
    expect(permanents(boardView(pay, PLAYERS)).every(item => item.paid === null)).toBe(true);
  });
});

describe('where decisions land', () => {
  it('names players in target choices as the board names them', () => {
    const bolt = (load(new URL('bolt-target.json', PROTOCOL)) as { recovery: { frame: ExperienceFrame } }).recovery.frame;
    const view = boardView(bolt, PLAYERS);
    expect(view.list.map(placed => placed.offer.label)).toEqual(['Target yourself', 'Target Heuristic']);
    expect(view.you.offers.map(placed => placed.offer.label)).toEqual(['Target yourself']);
    expect(view.opponent.offers.map(placed => placed.offer.label)).toEqual(['Target Heuristic']);
    // Watching a bot's seat names it, not "yourself"; seats in trigger labels are named too.
    const watched = boardView({ ...bolt, offers: [...bolt.offers, { ...bolt.offers[0], id: 2, label: 'Put on stack next — X — event: Seat 2' }] },
      { you: 'Search 64', opponent: 'Heuristic' });
    expect(watched.list.map(placed => placed.offer.label)).toEqual(['Target Search 64', 'Target Heuristic', 'Put on stack next — X — event: Heuristic']);
  });

  it('puts targets on the permanents they target', () => {
    const view = boardView(navigation.CHOOSE_TARGET.frame, PLAYERS);
    const targeted = permanents(view).filter(item => item.offers.length);
    expect(targeted.map(item => item.permanent.id).sort()).toEqual(
      navigation.CHOOSE_TARGET.frame.offers.map(offer => offer.focus[0]).sort());
    expect(view.tray).toEqual([]);
  });

  it('puts payment taps on lands and the payment on the primary button', () => {
    const view = boardView(navigation.PAY_OR_NOT.frame, PLAYERS);
    const lands = view.you.support.flatMap(pile => pile.items).filter(item => item.offers.length);
    expect(lands.flatMap(item => item.offers.map(placed => placed.offer.label)))
      .toEqual(expect.arrayContaining(['Tap Mountain for {R}', 'Tap Island for {U}']));
    expect(view.primary?.offer.label).toBe('Pay the cost');
  });

  it('draws Learn: discards on hand cards, Lessons in the tray by name, declining as a button', () => {
    const view = boardView(navigation.LEARN.frame, PLAYERS);
    expect(view.hand.filter(item => item.offers.length).length).toBe(6);
    expect(view.tray.map(item => item.name)).toEqual(['Accumulate Wisdom', 'Firebending Lesson', "It'll Quench Ya!"]);
    expect(view.buttons.map(placed => placed.offer.label)).toEqual(['Decline Learn']);
    // A click on the Learn spell on the stack never declines.
    expect(view.stack.flatMap(item => item.offers)).toEqual([]);
  });

  it('inspects tray cards as themselves: stash counters, a condition, a controlled opponent\'s hand', () => {
    const frame = structuredClone(navigation.PAY_OR_NOT.frame);
    const { agent, opponent } = frame.projection;
    const base = { registry_key: 0, power: 0, toughness: 0, mana_value: 1, types: agent.hand[0]?.types ?? ({} as never) };
    const stashed = { ...base, id: 9001, name: 'Lightning Bolt', zone: 'EXILE', owner_id: opponent.id, text_box: 'Deal 3.',
      counters: { stash: 1 }, cast_permission: { player_id: agent.id, cost: '{R}', own_turn_only: true, any_mana: true } };
    const flameshaped = { ...base, id: 9002, name: 'Mountain', zone: 'EXILE', owner_id: agent.id, face_down: true,
      cast_permission: { player_id: agent.id, face_down: true, condition: 'you control a Wizard' } };
    const controlled = { ...base, id: 9003, name: 'Island', zone: 'HAND', owner_id: opponent.id };
    opponent.exile = [...opponent.exile, stashed];
    agent.exile = [...agent.exile, flameshaped];
    opponent.inspected = [controlled];
    const offer = frame.offers.find(candidate => !['pass_priority', 'pay'].includes(candidate.verb))!;
    frame.offers = [9001, 9002, 9003].map((id, index) => ({ ...offer, id: index + 1, subjects: [], focus: [id], label: `Play ${id}` }));
    const view = boardView(frame, PLAYERS);
    expect(view.tray.map(item => [item.name, item.card?.id])).toEqual([
      ['Lightning Bolt', 9001], ['Mountain', 9002], ['Island', 9003]]);
    const player = (id: number) => id === agent.id ? 'You' : 'Heuristic';
    expect(counterTerms(view.tray[0].card?.counters)).toEqual(['1 stash counter']);
    expect(permissionTerms(view.tray[0].card, () => 'source', player)[0]).toContain('spending mana of any type');
    expect(permissionTerms(view.tray[1].card, () => 'source', player)[0]).toContain('only if you control a Wizard');
    expect(view.tray[2].card?.owner_id).toBe(opponent.id);
  });

  it('asks about one blocker: blocks on attackers, not blocking as a button', () => {
    const view = boardView(navigation.DECLARE_BLOCKER.frame, PLAYERS);
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
    const view = boardView(frame, PLAYERS);
    const attacker = permanents(view).find(item => item.permanent.id === creature)!;
    expect(attacker.offers.map(placed => placed.offer.label)).toEqual([frame.offers[0].label]);
    expect(view.buttons.map(placed => placed.offer.label)).toEqual([frame.offers[1].label]);
    expect(view.asked.has(creature)).toBe(true);
  });

  it('keeps combat damage amounts as buttons and asks about the attacker and blocker', () => {
    const frame = structuredClone(navigation.DECLARE_BLOCKER.frame);
    const [blocker, attacker] = frame.offers[0].subjects!.map(subject => subject.object);
    frame.action_space = 'MODAL';
    frame.projection.decision_context = { ...frame.projection.decision_context!, stage: 'assign_combat_damage', source: null };
    frame.offers = [1, 2].map((amount, id) => ({ ...frame.offers[0], id, verb: 'choose', action_type: 'MODAL_CHOOSE_MODE',
      label: `Attacker assigns ${amount} to Blocker`, focus: [attacker.display_id, blocker.display_id],
      subjects: [attacker, blocker].map(object => ({ role: 'choice', object })) }));
    const view = boardView(frame, PLAYERS);
    expect(view.buttons.map(placed => placed.offer.label)).toEqual(['Attacker assigns 1 to Blocker', 'Attacker assigns 2 to Blocker']);
    expect(permanents(view).flatMap(item => item.offers)).toEqual([]);
    expect(view.asked).toEqual(new Set([attacker.display_id, blocker.display_id]));
  });

  it('takes each land the payer would tap out of its pile', () => {
    const view = boardView(navigation.CHOOSE_TARGET.frame, PLAYERS);
    const pile = [...view.you.support].find(candidate => candidate.items.length > 2)!;
    const planned = new Map([[pile.items[0].permanent.id, 1]]);
    const split = unpilePlanned([pile], planned);
    expect(split.map(candidate => candidate.items.length)).toEqual([1, pile.items.length - 1]);
    expect(split[0].items[0].permanent.id).toBe(pile.items[0].permanent.id);
    expect(unpilePlanned([pile], new Map())).toEqual([pile]);
  });

  it('keeps permanents with different loyalty activation histories separately inspectable', () => {
    const frame = structuredClone(navigation.CHOOSE_TARGET.frame);
    const agent = frame.projection.agent;
    const token = { ...agent.battlefield[0], id: 9001, name: 'Jace', printed_loyalty: 0, counters: { loyalty: 2 } };
    agent.battlefield = [token, { ...token, id: 9002, loyalty_activated_this_turn: true }];
    frame.offers = [];
    const side = boardView(frame, PLAYERS).you;
    expect([...side.support, ...side.creatures].map(pile => pile.items.length)).toEqual([1, 1]);
  });

  it('fans identical lands into one pile, keeping each copy', () => {
    const view = boardView(navigation.CHOOSE_TARGET.frame, PLAYERS);
    const piles = [...view.you.support, ...view.you.creatures];
    expect(piles.some(pile => pile.items.length > 1)).toBe(true);
    expect(piles.flatMap(pile => pile.items).length).toBe(navigation.CHOOSE_TARGET.frame.projection.agent.battlefield.length);
  });
});

describe('fitRow', () => {
  const limits = { gap: 8, minCard: 44, maxCard: 140, maxLines: 3 };
  const pile = (n: number, tapped = false): PileShape => Array(n).fill(tapped);
  const upright = (n: number): PileShape[] => Array.from({ length: n }, () => pile(1));

  it('uses the full height for a small board, at a printed card\'s proportions', () => {
    const fit = fitRow(upright(3), 800, 150, limits);
    expect(fit.lines).toBe(1);
    expect(fit.card).toBe(Math.floor(150 * 63 / 88));
  });

  it('reserves a tapped card\'s sideways width, so tapping can shrink a full row', () => {
    const width = 3 * 100 + 2 * 8;
    expect(fitRow(upright(3), width, 140, limits).card).toBe(100);
    const tapped = fitRow([pile(1, true), pile(1), pile(1)], width, 140, limits);
    expect(tapped.card).toBeLessThan(100);
    expect(footprint(tapped.card, true) + 2 * tapped.card + 2 * 8).toBeLessThanOrEqual(width);
  });

  it('measures a pile by its widest reach', () => {
    expect(pileWidth(pile(3), 100, 22)).toBe(144);
    expect(pileWidth(pile(3, true), 100, 22)).toBe(44 + footprint(100, true));
    // An untapped Aura on a tapped creature: the creature lies wider than the Aura's offset.
    expect(pileWidth([true, false], 100, 22)).toBe(footprint(100, true));
  });

  it('keeps twenty-four distinct creatures readable by wrapping', () => {
    const fit = fitRow(upright(24), 820, 300, limits);
    expect(fit.overflow).toBe(false);
    expect(fit.card).toBeGreaterThanOrEqual(55);
    expect(fit.lines).toBeGreaterThan(1);
  });

  it('fits a pile of twelve lands in less width than twelve cards', () => {
    const piled = fitRow([pile(12)], 400, 120, limits);
    const spread = fitRow(upright(12), 400, 120, limits);
    expect(piled.card).toBeGreaterThan(spread.card);
  });

  it('reports overflow only past the smallest size', () => {
    expect(fitRow(upright(80), 600, 100, limits).overflow).toBe(true);
  });

  it('finds the same largest card as trying every size', () => {
    // Every card size from the largest down, as a person would by eye: the first that fits wins.
    const walk = (piles: PileShape[], width: number, height: number) => {
      let best: { card: number; lines: number } | null = null;
      for (let lines = 1; lines <= limits.maxLines; lines += 1) {
        const lineHeight = (height - limits.gap * (lines - 1)) / lines;
        for (let card = Math.min(limits.maxCard, Math.floor(lineHeight * 63 / 88)); card >= limits.minCard; card -= 1) {
          const fan = Math.max(8, Math.round(card * 0.22));
          let used = 0, needed = 1;
          for (const shape of piles) {
            const w = Math.max(...shape.map((turned, index) => index * fan + (turned ? Math.round(card * 88 / 63) : card)));
            if (used > 0 && used + limits.gap + w > width) { needed += 1; used = w; } else used += (used > 0 ? limits.gap : 0) + w;
          }
          if (needed <= lines) { if (!best || card > best.card) best = { card, lines }; break; }
        }
      }
      return best;
    };
    let seed = 7;
    const random = (n: number) => { seed = (seed * 1103515245 + 12345) % 2 ** 31; return seed % n; };
    for (let trial = 0; trial < 400; trial += 1) {
      const piles = Array.from({ length: 1 + random(30) }, () => pile(1 + (random(4) === 0 ? random(9) : 0), random(3) === 0));
      const width = 200 + random(1000), height = 40 + random(400);
      const fit = fitRow(piles, width, height, limits);
      const expected = walk(piles, width, height);
      expect(fit.overflow ? null : { card: fit.card, lines: fit.lines }).toEqual(expected);
    }
  });
});

describe('splitHalf', () => {
  const limits = { gap: 6, creatures: { gap: 8, minCard: 40, maxCard: 132, maxLines: 3 }, support: { gap: 8, minCard: 36, maxCard: 104, maxLines: 2 } };
  const upright = (n: number): PileShape[] => Array.from({ length: n }, () => [false]);

  it('gives a crowded creature row the height a few land piles do not need', () => {
    const split = splitHalf(upright(18), [Array(5).fill(false), Array(5).fill(false)], 840, 280, limits);
    expect(split.creatures).toBeGreaterThan(280 * 0.58);
    expect(split.creatureFit.card).toBeGreaterThanOrEqual(60);
    expect(split.creatureFit.overflow || split.supportFit.overflow).toBe(false);
  });

  it('keeps lands readable when there are no creatures', () => {
    const split = splitHalf([], upright(8), 840, 280, limits);
    expect(split.supportFit.card).toBeGreaterThan(80);
  });
});

describe('the spell being cast', () => {
  it('is on top of the stack while it is paid for, not in the hand', () => {
    const frame = navigation.PAY_OR_NOT.frame;
    const source = frame.projection.decision_context!.source!;
    expect(frame.projection.decision_context!.announcing).toBe('spell');
    const view = boardView(frame, PLAYERS);
    expect(view.stack[0].announcing).toBe('spell');
    expect(view.stack[0].card?.id).toBe(source.display_id);
    expect(view.stack[0].entry.label).toBe(source.label);
    expect(view.stack.length).toBe((frame.projection.stack_objects ?? []).length + 1);
    expect(view.hand.map(item => item.card.id)).not.toContain(source.display_id);
  });

  it('draws a spell announced from outside the hand with its card', () => {
    const frame = structuredClone(navigation.PAY_OR_NOT.frame);
    const source = frame.projection.decision_context!.source!;
    const agent = frame.projection.agent;
    const card = agent.hand.find(candidate => candidate.id === source.display_id)!;
    agent.hand = agent.hand.filter(candidate => candidate !== card);
    agent.graveyard = [...agent.graveyard, { ...card, zone: 'graveyard' }];
    const view = boardView(frame, PLAYERS);
    expect(view.stack[0]).toMatchObject({ announcing: 'spell', card: { id: source.display_id } });
  });

  it('names an ability being activated by its source and leaves the source on the battlefield', () => {
    const frame = structuredClone(navigation.PAY_OR_NOT.frame);
    const land = frame.projection.agent.battlefield[0];
    const context = frame.projection.decision_context!;
    context.announcing = 'activated_ability';
    context.source = { entity_id: land.id, subject: { kind: 'object', id: { entity: land.id, incarnation: 0 } }, display_id: land.id, label: land.name ?? "Land" };
    const view = boardView(frame, PLAYERS);
    expect(view.stack[0]).toMatchObject({ announcing: 'activated_ability', card: null, entry: { source_permanent: land.id } });
    expect(permanents(view).map(item => item.permanent.id)).toContain(land.id);
  });
});

describe('handStrip', () => {
  const limits = {
    gap: 6,
    creatures: { gap: 8, minCard: 48, maxCard: 140, maxLines: 3 },
    support: { gap: 6, minCard: 40, maxCard: 104, maxLines: 2 },
  };
  // The layout's range on a crowded board: its rows may take the hand down from 232 to 160.
  const range = { min: 160, max: 232, step: 8 };
  const half = (creatures: number, lands: number) => ({
    creatures: Array.from({ length: creatures }, () => [false]), support: Array.from({ length: lands }, () => [false]), width: 830 });

  it('keeps a crowded board\'s height for the board', () => {
    expect(handStrip(720, [half(18, 12), half(18, 12)], limits, range)).toBeLessThanOrEqual(176);
  });

  it('never shrinks any row by more than 3% of its card width', () => {
    const halves = [half(7, 7), half(8, 9)];
    const hand = handStrip(720, halves, limits, range);
    for (const h of halves) {
      const at = (strip: number) => splitHalf(h.creatures, h.support, h.width - 22, (720 - strip) / 2 - 17, limits);
      expect(at(hand).creatureFit.card).toBeGreaterThanOrEqual(at(range.min).creatureFit.card * 0.97);
      expect(at(hand).supportFit.card).toBeGreaterThanOrEqual(at(range.min).supportFit.card * 0.97);
    }
  });
});

describe('names in sentences', () => {
  it('drops the kind and takes a plain apostrophe after a final s', () => {
    expect(shortName('Heuristic · scripted')).toBe('Heuristic');
    expect(possessive(shortName('Heuristic · scripted'))).toBe("Heuristic's");
    expect(possessive('Invasion Reinforcements')).toBe("Invasion Reinforcements'");
    expect(possessive('You')).toBe('your');
  });
});

describe('stack provenance', () => {
  const spell = { id: 1, kind: 'spell', controller: 0, label: 'Soul Tether' };
  it('tells a prepared spell from its copy and names flashback', () => {
    expect(stackProvenance({ ...spell, from_prepared: true })).toEqual(['prepared']);
    expect(stackProvenance(spell)).toEqual([]);
    expect(stackProvenance({ ...spell, flashback: true })).toEqual(['flashback']);
  });
});
