<script lang="ts">
  /**
   * One card on the board: a permanent, a card in hand, a spell on the stack
   * or a tray candidate. It is a button whenever the viewer may inspect it;
   * its outline says a legal offer acts on it, and its accessible name says
   * which.
   *
   * Both frames are a printed card's proportions (63:88). `portrait` (hand,
   * tray) is art over a bar with the name and type line, cost pips on the
   * art. `field` (battlefield) puts the name bar on top, wrapping to a second
   * line rather than cutting the name, a pile's ×N right after it, with the
   * art below and no cost (the hand and inspect show it). The name bar
   * carries the card's colours (`frameColors`): one colour's tint, or several
   * blended left to right as printed hybrid frames are. A tapped permanent
   * lies on its side: its footprint and art turn a quarter, while the name
   * bar stays across the top so it still reads. State chips (counters,
   * keywords, summoning sickness, attacking, blocking, damage) sit on their
   * own row above power and toughness. Rules text lives in the inspect view.
   *
   * Lift + Quiet: a card a legal offer acts on is raised with a soft shadow;
   * the selected card rises higher with a thin ink edge. While the person
   * may act, what they cannot act on steps back by a filter, never opacity,
   * so piles stay solid: an uncastable hand card most (`dim="hand"`), an
   * idle permanent of theirs slightly (`dim="idle"`). An attacker steps
   * toward its target and tilts a few degrees; a target of something on the
   * stack has a dashed edge in the damage colour.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import { cardFace, frameColors } from '$lib/card-faces.svelte';
  import { labelText } from './inspect';
  import type { CardState, PermanentState } from '$lib/types';
  import { getContext } from 'svelte';
  import { keywordsOf } from './inspect';
  import { cardEvidence, EVIDENCE, formatProbability, type OfferEvidence } from './replay';
  import type { PlannedSource } from './payment';
  import Pips from './Pips.svelte';
  import { footprint, type PlacedOffer } from './view';

  interface Props {
    name: string;
    /** Upright width, px; a tapped permanent's footprint is this card on its side. */
    width: number;
    shape?: 'portrait' | 'field';
    permanent?: PermanentState | null;
    card?: CardState | null;
    offers?: PlacedOffer[];
    /** Offers may be committed now. */
    live?: boolean;
    connectSource?: boolean;
    connectTarget?: boolean;
    drop?: boolean;
    asked?: boolean;
    /** The side of the centre line this card sits on; attacks lean toward it. */
    toward?: 'up' | 'down' | null;
    /** The name of the attacker this permanent blocks; the line on the board draws it. */
    blocks?: string | null;
    /** Whose plate a card new to the board travels from. */
    side?: 'you' | 'opponent';
    /** It attacked, blocked, was dealt damage or was targeted since your last move. */
    touched?: boolean;
    /** The opponent's move being read out is about this card. */
    beat?: boolean;
    /** Damage it was dealt since your last move, shown as it lands. */
    dealt?: number;
    /** Its place in the automatic payer's plan: "would tap, for {U}". */
    planned?: PlannedSource | null;
    /** What it has committed to the pending cost: "tapped to crew". */
    paid?: string | null;
    /** Copies in this card's pile, shown after the top copy's name. */
    count?: number | null;
    /** A copy beneath its pile's top card: the top card's bar names the pile, so this one's name stays hidden. */
    piled?: boolean;
    /**
     * How much of a covered card in hand stays in view, from its left edge,
     * px: its cost moves from the art into the name bar, before the name, so
     * the visible slice reads both.
     */
    slice?: number | null;
    /** While the person may act, this card steps back: an uncastable hand card (`hand`) or an idle permanent (`idle`). */
    dim?: 'hand' | 'idle' | null;
    /** Something on the stack targets it. */
    targeted?: boolean;
    /** Its place in a combat draft: it may join (`can`), it is drafted (`in`), or it is the blocker being placed (`armed`). */
    draft?: 'can' | 'in' | 'armed' | null;
    /** What pressing it does in a combat draft, for its accessible name. */
    note?: string | null;
    /** It takes part in travel between frames; a card in hand being carried leaves that to the carried copy. */
    travel?: boolean;
    onactivate?: (element: HTMLElement) => void;
    onpreview?: (element: HTMLElement | null) => void;
  }

  let {
    connectSource = false, connectTarget = false, drop = false, name, width, shape = 'portrait', permanent = null, card = null, offers = [], live = false, asked = false,
    toward = null, blocks = null, side = 'you', touched = false, beat = false, dealt = 0, planned = null, paid = null,
    count = null, piled = false, slice = null, dim = null, targeted = false, draft = null, note = null, travel = true, onactivate, onpreview,
  }: Props = $props();

  // At a recorded choice: the policy's mass on this card's offers, and whether one was played.
  const recordedPolicy = getContext<(() => OfferEvidence | null) | undefined>(EVIDENCE);
  const recorded = $derived(cardEvidence(recordedPolicy?.() ?? null, offers.map(placed => placed.offer.id)));

  const field = $derived(shape === 'field');
  const tapped = $derived(permanent?.tapped ?? false);
  const turned = $derived(field && tapped);
  const upright = $derived(footprint(width, true));
  // A turned card's box is the upright card on its side.
  const boxWidth = $derived(turned ? upright : width);
  const boxHeight = $derived(turned ? width : upright);
  const creature = $derived(permanent ? permanent.types?.is_creature ?? permanent.power !== null : card?.types.is_creature ?? false);
  const power = $derived(permanent?.power ?? card?.power ?? null);
  const toughness = $derived(permanent?.toughness ?? card?.toughness ?? null);
  const face = $derived(cardFace(permanent?.copy_of?.name ?? name));
  const keywords = $derived(permanent ? keywordsOf(permanent, face) : (face?.keywords ?? []).map(keyword => ({ keyword, source: null })));
  const counters = $derived(Object.entries({
    ...(permanent?.plus1_counters ? { '+1/+1': permanent.plus1_counters } : {}),
    ...(permanent?.counters ?? {}),
  }).filter(([, amount]) => amount > 0));

  // Type comes from the six sizes of DESIGN.md, each with its line height.
  // Battlefield names are 12 px at every size: a crowded row wraps them.
  const LINE: Record<number, number> = { 12: 16, 14: 20 };
  const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));
  const nameSize = $derived(field ? 12 : 14);
  const showType = $derived(!field && upright >= 110);
  const pip = $derived(clamp(upright * 0.07, 10, 14));
  const showCost = $derived(!field && width >= 58);
  const plate = $derived(upright >= 110 ? 14 : 12);
  const keywordRoom = $derived(boxWidth >= 120 ? 3 : width >= 64 ? 1 : 0);
  const hasPt = $derived(creature && power !== null);
  /**
   * State chips, on their own row above power and toughness. Combat and
   * damage come first: the row wraps upward and a small card keeps its
   * first chips, so keywords (also in inspect) give way before them.
   */
  const chips = $derived(!field ? [] : [
    ...(permanent?.first_strike_participant ? [{ key: 'first-strike-participant', text: 'First-strike participant', kind: 'combat', granted: false }] : []),
    ...(permanent?.attacking ? [{ key: 'attacking', text: 'Attacking', kind: 'combat', granted: false }] : []),
    ...(permanent?.blocking !== undefined ? [{ key: 'blocking', text: 'Blocking', kind: 'combat', granted: false }] : []),
    ...(permanent?.damage ? [{ key: 'damage', text: `${permanent.damage} damage`, kind: 'damage', granted: false }] : []),
    ...(permanent?.summoning_sick && creature ? [{ key: 'sick', text: width >= 100 ? 'Summoning sick' : 'Sick', kind: 'sick', granted: false }] : []),
    ...counters.map(([kind, amount]) => ({ key: `counter-${kind}`, text: kind === '+1/+1' ? `+${amount}/+${amount}` : `${labelText(kind)} ×${amount}`, kind: 'counter', granted: false })),
    ...keywords.slice(0, keywordRoom).map(fact => ({ key: `keyword-${fact.keyword}`, text: labelText(fact.keyword), kind: 'keyword', granted: fact.source !== null })),
    ...(keywords.length > keywordRoom && keywordRoom > 0 ? [{ key: 'keyword-more', text: `${keywords.length - keywordRoom} more`, kind: 'keyword', granted: false }] : []),
  ]);

  // The name bar's colour: one colour's tint, several blended left to right
  // (each holds its side, blended across the middle), colourless grey.
  const colors = $derived(frameColors(face).map(color => color.toLowerCase()));
  const barStyle = $derived.by(() => {
    const [first = 'c'] = colors;
    const ink = `--bar-ink: var(--card-${first}-ink)`;
    if (colors.length < 2) return `--bar: var(--card-${first}); ${ink}`;
    const band = 30 / colors.length;
    const stops = colors.map((color, index) => {
      const from = index === 0 ? 0 : (index * 100) / colors.length + band / 2;
      const to = index === colors.length - 1 ? 100 : ((index + 1) * 100) / colors.length - band / 2;
      return `var(--card-${color}) ${from}% ${to}%`;
    });
    return `--bar: linear-gradient(90deg, ${stops.join(', ')}); ${ink}`;
  });

  const url = $derived(artUrl(name));
  let loaded = $state<string | null>(null);
  $effect(() => {
    const requested = url;
    let active = true;
    void loadArt(requested, name).then(ok => { if (active && ok) loaded = requested; });
    return () => { active = false; };
  });
  const art = $derived(loaded === url ? url : null);

  const facts = $derived([
    tapped ? 'tapped' : '',
    permanent?.summoning_sick && creature ? 'summoning sick' : '',
    permanent?.attacking ? 'attacking' : '',
    permanent?.first_strike_participant ? 'first-strike participant' : '',
    permanent?.blocking !== undefined ? `blocking ${blocks ?? 'an attacker'}` : '',
    permanent?.damage ? `${permanent.damage} damage` : '',
    ...counters.map(([kind, amount]) => `${amount} ${kind} counter${amount === 1 ? '' : 's'}`),
    permanent?.is_token ? 'token' : '',
    dealt ? `dealt ${dealt} since your last move` : '',
    paid ? `committed to this cost: ${paid}` : '',
    planned ? `would tap for ${planned.output || 'mana'} (payment plan, source ${planned.order})` : '',
    recorded?.p != null ? `recorded policy ${formatProbability(recorded.p)}` : '',
    recorded?.chosen ? 'chosen' : '',
  ].filter(Boolean).join(', '));
  const label = $derived([
    `${name}${creature && power !== null ? ` ${power}/${toughness}` : ''}`,
    facts,
    note ?? '',
    offers.length === 1 ? `${offers[0].offer.label} (${offers[0].number})`
      : offers.length > 1 ? `${offers.length} choices: ${offers.map(placed => placed.offer.label).join('; ')}` : '',
  ].filter(Boolean).join(' — '));
</script>

<div
  class="slot"
  class:lean-up={permanent?.attacking && toward === 'up'}
  class:lean-down={permanent?.attacking && toward === 'down'}
  style:width={`${boxWidth}px`}
  style:height={`${boxHeight}px`}
>
  <button
    type="button"
    class="tcard"
    class:field
    class:turned
    class:legal={live && (offers.length > 0 || draft === 'can')}
    class:dim-hand={dim === 'hand'}
    class:dim-idle={dim === 'idle'}
    class:targeted
    class:downward={toward === 'down'}
    class:drafted={draft === 'in'}
    class:armed={draft === 'armed'}
    class:offered={!live && recorded !== null}
    class:chosen={recorded?.chosen}
    class:asked
    class:piled
    class:attacking={permanent?.attacking}
    class:touched
    class:beat
    class:planned={planned !== null}
    class:sliced={slice !== null}
    class:paid={paid !== null}
    data-connect-source={connectSource || undefined} data-connect-target={connectTarget || undefined} data-drop={drop || undefined}
    data-testid="board-card"
    data-travel-id={!travel ? undefined : permanent ? `perm:${permanent.id}` : card ? `card:${card.id}` : undefined}
    data-travel-name={travel ? name : undefined}
    data-travel-side={side}
    data-object-id={permanent?.id ?? card?.id}
    data-offer-count={offers.length}
    data-draft={draft ?? undefined}
    data-payment-source={planned?.order}
    data-paid={paid ?? undefined}
    data-card-name={name}
    aria-label={label}
    style:width={`${boxWidth}px`}
    style:height={`${boxHeight}px`}
    onclick={event => onactivate?.(event.currentTarget)}
    onmouseenter={event => onpreview?.(event.currentTarget)}
    onmouseleave={() => onpreview?.(null)}
    onfocus={event => onpreview?.(event.currentTarget)}
    onblur={() => onpreview?.(null)}
  >
    <span class="bar" aria-hidden="true" style={barStyle} style:--slice={slice === null ? undefined : `${slice - 10}px`}>
      <span class="title">
        {#if slice !== null && face?.cost && !showType}<span class="bar-cost" data-testid="slice-cost"><Pips cost={face.cost} size={pip} /></span>{/if}
        <span class="name" style:font-size={`${nameSize}px`} style:line-height={`${LINE[nameSize]}px`}>{name}{#if count}{' '}<b class="count" data-testid="pile-count-badge">×{count}</b>{/if}</span>
      </span>
      {#if showType && face}
        {#if slice !== null && face.cost}<span class="bar-cost on-type" data-testid="slice-cost"><Pips cost={face.cost} size={pip} /></span>{/if}
        <span class="type">{face.type_line}</span>
      {/if}
    </span>
    <span class="art" aria-hidden="true">
      <span class="picture" style:background-image={art ? `url('${art}')` : undefined}>
        {#if !art}<span class="no-art">{face?.type_line ?? ''}</span>{/if}
      </span>
      <span class="corner">
        {#if showCost && face?.cost}<span class="cost"><Pips cost={face.cost} size={pip} /></span>{/if}
        {#if offers.length > 1 && live}<span class="many">{offers.length}</span>{/if}
      </span>
      {#if chips.length}
        <span class="chips" style:--below={`${hasPt ? LINE[plate] + 4 : 3}px`}>
          {#each chips as chip (chip.key)}<i class={`chip ${chip.kind}`} class:granted={chip.granted}>{chip.text}</i>{/each}
        </span>
      {/if}
      {#if hasPt}
        <span class="pt" class:hurt={(permanent?.damage ?? 0) > 0}
          class:up={permanent && ((permanent.power ?? 0) > (face?.power ?? Infinity) || (permanent.toughness ?? 0) > (face?.toughness ?? Infinity))}
          style:font-size={`${plate}px`} style:line-height={`${LINE[plate]}px`}>{power}/{toughness}</span>
      {/if}
      {#if planned}
        <span class="plan"><b>{planned.order}</b>{#if planned.output}<Pips cost={planned.output} size={Math.max(10, pip)} />{/if}</span>
      {/if}
      {#if paid}<span class="paid-mark" title={paid}>committed</span>{/if}
      {#if recorded && (recorded.p !== null || recorded.chosen)}
        <span class="policy" data-testid="card-policy" data-chosen={recorded.chosen}>{#if recorded.chosen}✓{/if}{recorded.p !== null ? formatProbability(recorded.p) : ''}</span>
      {/if}
      {#if dealt}{#key dealt}<span class="dealt" data-testid="damage-dealt">−{dealt}</span>{/key}{/if}
    </span>
  </button>
</div>

<style>
  .slot { position: relative; flex: none; transition: transform var(--motion-move) var(--motion-move-ease); }
  /* An attacker steps toward the player it attacks by `--lean`, which its row sizes (Field). */
  .slot.lean-up { transform: translateY(calc(-1 * var(--lean, 20%))); }
  .slot.lean-down { transform: translateY(var(--lean, 20%)); }

  /* Rise and tilt compose: a raised card may also be an attacker. */
  .tcard {
    --rise: 0px; --tilt: 0deg;
    position: relative; display: flex; flex-direction: column-reverse; overflow: hidden; padding: 0; cursor: pointer; text-align: left;
    border-radius: var(--radius-card); border: 1px solid var(--border);
    background: var(--bg-surface); color: var(--text);
    transform: translateY(var(--rise)) rotate(var(--tilt));
    transition: transform var(--motion-move) var(--motion-move-ease), box-shadow var(--motion-move) ease-out, filter var(--motion-move) ease-out;
  }
  .tcard:focus-visible { outline: 2px solid var(--text); outline-offset: 3px; }
  .turned { border-radius: var(--radius-card-turned); }

  /* The name bar: under the art in hand, across the top on the battlefield, upright either way, in the card's colours. */
  .bar {
    flex: none; min-width: 0; padding: 3px 5px 4px; display: flex; flex-direction: column; gap: 1px;
    background: var(--bar, var(--card-c)); color: var(--bar-ink, var(--text));
  }
  .field { flex-direction: column; }
  .field .bar { padding: 2px 4px 3px; }
  .name {
    font-family: var(--font-serif); font-weight: 600; color: inherit;
    overflow: hidden; overflow-wrap: break-word; hyphens: auto; text-wrap: balance;
    display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2;
  }
  /* In hand a name holds one line, shortened (Inspect has it whole). A covered card's slice gives the name its
     whole width and shows the cost in place of the type line; on a card too small for a type line the cost leads the name. */
  .title { display: flex; align-items: center; gap: 3px; min-width: 0; }
  .bar-cost { flex: none; display: flex; }
  .field .title { display: block; }
  .tcard:not(.field) .name { display: block; flex: 1 1 auto; min-width: 0; white-space: nowrap; text-overflow: ellipsis; text-wrap: nowrap; }
  /* Lifted by pointer or focus, a covered card shows whole, as it is laid out uncovered. */
  .sliced:not(:hover, :focus-visible) :is(.title, .type) { max-width: var(--slice); }
  .sliced:not(:hover, :focus-visible) :is(.corner .cost, .on-type + .type), .sliced:is(:hover, :focus-visible) .bar-cost { display: none; }
  .bar-cost.on-type { height: 16px; align-items: center; }
  /* A pile's count follows the top copy's name, in the interface face; the line breaks before it, never inside the
     name. The copies beneath keep their bar's height but show no name. */
  .piled .title { visibility: hidden; }
  .count { font-family: var(--font-sans); font-weight: 600; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .type { font: 12px/16px var(--font-sans); color: inherit; opacity: 0.78; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* The art fills what the bar leaves. A turned card's picture is the upright art turned a quarter. */
  .art { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; container-type: size; background: var(--bg-muted); }
  .picture { position: absolute; inset: 0; background: var(--bg-muted) center 30% / cover no-repeat; }
  .turned .picture {
    inset: auto; left: 50%; top: 50%; width: 100cqh; height: 100cqw;
    transform: translate(-50%, -50%) rotate(90deg);
  }
  .no-art { position: absolute; inset: 4px; display: grid; place-items: center; text-align: center; font: 12px/16px var(--font-sans); color: var(--text-tertiary); overflow: hidden; }

  .corner { position: absolute; top: 3px; right: 3px; display: flex; gap: 3px; align-items: flex-start; }
  .cost { padding: 1px 2px; border-radius: var(--radius-full); background: color-mix(in srgb, var(--bg-surface) 85%, transparent); display: flex; }
  .many {
    min-width: 16px; height: 16px; border-radius: var(--radius-full); padding: 0 4px;
    font: 600 12px/16px var(--font-sans); text-align: center;
    color: var(--on-accent); background: var(--accent);
  }

  /* Chips: one row above power and toughness, wrapping upward, never under the plate; only whole rows that fit the art show. */
  .chips {
    position: absolute; left: 3px; right: 3px; bottom: var(--below); display: flex; flex-wrap: wrap-reverse; align-content: flex-start; gap: 2px; overflow: hidden;
    max-height: round(down, calc(100cqh - var(--below) + 2px), 18px);
  }
  .chip {
    max-width: 100%; padding: 0 5px; border-radius: var(--radius-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font: 400 12px/16px var(--font-sans);
    color: var(--text); background: var(--bg-surface); box-shadow: 0 0 0 1px var(--border);
  }
  .chip.combat, .chip.counter { font-weight: 600; }
  .chip.damage { font-weight: 600; color: var(--damage); }
  .chip.granted { box-shadow: 0 0 0 1px var(--gain); }

  /* An attacker tilts a few degrees past its rest: a tapped one lies just past the quarter, a vigilant one stays upright. */
  .tcard.attacking { --tilt: -2deg; box-shadow: var(--lift-attacking); }
  .tcard.attacking.turned { --tilt: -6deg; }
  .tcard.attacking.downward { --tilt: 2deg; }
  .tcard.attacking.turned.downward { --tilt: 6deg; }
  /* Lift: what a legal offer acts on is raised; the selected card higher, with an ink edge. */
  .tcard.legal { --rise: -6px; box-shadow: var(--lift); }
  .tcard.legal:hover { --rise: -8px; }
  .tcard.asked, .tcard.armed { --rise: -12px; box-shadow: var(--lift-selected); outline: 2px solid var(--accent); outline-offset: 0; }
  /* Quiet: what cannot act steps back by a filter, never opacity, so a pile stays solid. */
  .tcard.dim-hand { filter: var(--dim-hand); }
  .tcard.dim-idle { filter: var(--dim-idle); }
  .tcard.targeted { outline: 2px dashed var(--damage); outline-offset: 2px; }
  .tcard.touched { box-shadow: 0 0 0 2px color-mix(in srgb, var(--text) 40%, transparent); }
  /* A recorded choice's legal space, read-only: offered cards are outlined thinly, the played one solidly. */
  .tcard.offered { box-shadow: 0 0 0 1px color-mix(in srgb, var(--text) 55%, transparent); }
  .tcard.chosen { box-shadow: 0 0 0 2px var(--text), var(--lift); }
  .policy {
    position: absolute; left: 50%; bottom: 3px; transform: translateX(-50%); padding: 0 5px; border-radius: var(--radius-full);
    font: 600 12px/16px var(--font-sans); font-variant-numeric: tabular-nums; white-space: nowrap;
    color: var(--text); background: var(--bg-surface); border: 1px solid var(--border-strong);
  }
  .policy[data-chosen='true'] { color: var(--bg-surface); background: var(--text); border-color: var(--text); }
  .tcard.beat { box-shadow: 0 0 0 3px var(--text), var(--lift-selected); transition: box-shadow 160ms ease-out; }
  /* Drafted into an attack or block: a dashed ink ring, distinct from a declared attacker. */
  .tcard.drafted { outline: 2px dashed var(--accent); outline-offset: 2px; }
  .tcard.armed { outline-style: dashed; outline-offset: 2px; }

  /* Planned payment: dashed amber, and what tapping it adds. */
  .tcard.planned { outline: 2px dashed var(--planned); outline-offset: 2px; }
  .plan {
    position: absolute; left: 3px; top: 3px;
    display: inline-flex; align-items: center; gap: 3px; padding: 1px 4px 1px 3px; border-radius: var(--radius-full);
    background: var(--bg-surface); border: 1px dashed var(--planned);
  }
  /* Committed to the pending cost: solid amber, where a plan is dashed. */
  .tcard.paid { outline: 2px solid var(--planned); outline-offset: 2px; }
  .paid-mark {
    position: absolute; left: 50%; top: 3px; transform: translateX(-50%);
    padding: 0 5px; border-radius: var(--radius-full);
    background: var(--bg-surface); border: 1px solid var(--planned);
    font: 600 12px/16px var(--font-sans); color: var(--planned-text);
  }
  .plan b { font: 600 12px/16px var(--font-sans); color: var(--planned-text); min-width: 9px; text-align: center; }
  .dealt {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%, -30%);
    font: 600 20px/28px var(--font-sans); color: var(--bg);
    padding: 0 6px; border-radius: var(--radius-sm); background: var(--damage); box-shadow: var(--shadow-raised);
    animation: land calc(var(--motion-move) * 1.6) var(--motion-move-ease);
  }
  @keyframes land { from { transform: translate(-50%, -90%) scale(1.25); opacity: 0; } }
  .pt {
    position: absolute; right: 3px; bottom: 3px; padding: 0 0.35em; border-radius: var(--radius-sm);
    font-family: var(--font-sans); font-weight: 600; font-variant-numeric: tabular-nums;
    color: var(--text); background: var(--bg-surface); box-shadow: 0 0 0 1px var(--border);
  }
  /* A covered card's power and toughness sit in the slice that shows. */
  .sliced:not(:hover, :focus-visible) .pt { right: auto; left: 3px; }
  .pt.up { color: var(--gain); }
  .pt.hurt { color: var(--damage); }
  @media (prefers-reduced-motion: reduce) {
    .slot, .tcard { transition: none; }
    .dealt { animation: none; }
  }
</style>
