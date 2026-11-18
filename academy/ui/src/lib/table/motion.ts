/**
 * Cards move between zones instead of jumping.
 *
 * `Travel` remembers where every card was drawn just before a frame changes
 * the table (`snapshot`), then animates each card from there to where it is
 * drawn now (`play`): a card that stayed in place but moved because its row
 * reflowed slides; a card that is new to the table comes from the card of the
 * same name that has just left (a spell from hand to the stack, a permanent
 * from the stack to the battlefield), or else from its owner's plate, as an
 * opponent's card arrives from a hidden hand. This is a rendering of state
 * the frame already shows: identities stay the authority's, names only guide
 * the eye, and with reduced motion nothing animates and nothing is lost.
 *
 * Elements opt in with `data-travel-id` (the object or card id), and
 * `data-travel-name`; `data-travel-side` ("you" | "opponent") names whose plate a
 * card with no earlier position comes from. The animation uses the CSS
 * `translate` and `scale` properties, which compose with a card's own
 * `transform` (tapped rotation, lift) rather than replacing it.
 */

interface Seen {
  id: string;
  name: string;
  rect: DOMRect;
}

const MOVE_MS = 240;
const ARRIVE_MS = 320;
const EASE = 'cubic-bezier(0.2, 0.7, 0.2, 1)';

function cards(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-travel-id]')];
}

function visible(rect: DOMRect): boolean {
  return rect.width > 0 && rect.height > 0;
}

export class Travel {
  private seen: Seen[] = [];

  /** Records where every travelling card is drawn now; call before the DOM updates. */
  snapshot(root: HTMLElement | null): void {
    this.seen = root ? cards(root).flatMap(element => {
      const rect = element.getBoundingClientRect();
      return visible(rect) ? [{ id: element.dataset.travelId!, name: element.dataset.travelName ?? '', rect }] : [];
    }) : [];
  }

  /** Animates every card from its snapshot position, or from where it came from, to where it is now. */
  play(root: HTMLElement | null, reduced: boolean): void {
    const before = this.seen;
    this.seen = [];
    if (!root || reduced || !before.length) return;
    const now = cards(root);
    const present = new Set(now.map(element => element.dataset.travelId!));
    const byId = new Map(before.map(seen => [seen.id, seen]));
    // Cards that left the table this frame, by name, for new cards to come from.
    const departed = new Map<string, Seen[]>();
    for (const seen of before) {
      if (present.has(seen.id)) continue;
      departed.set(seen.name, [...(departed.get(seen.name) ?? []), seen]);
    }
    const plates = new Map([...root.querySelectorAll<HTMLElement>('[data-plate-side]')]
      .map(plate => [plate.dataset.plateSide!, plate.getBoundingClientRect()]));
    for (const element of now) {
      const rect = element.getBoundingClientRect();
      if (!visible(rect)) continue;
      const id = element.dataset.travelId!;
      const stayed = byId.get(id);
      if (stayed) {
        if (Math.abs(stayed.rect.left - rect.left) > 1 || Math.abs(stayed.rect.top - rect.top) > 1) {
          move(element, stayed.rect, rect, MOVE_MS, false);
        }
        continue;
      }
      const source = departed.get(element.dataset.travelName ?? '')?.shift();
      const plate = plates.get(element.dataset.travelSide ?? '');
      if (source) move(element, source.rect, rect, ARRIVE_MS, false);
      else if (plate && element.dataset.travelSide === 'opponent') move(element, plate, rect, ARRIVE_MS, true);
      else element.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 180, easing: 'ease-out' });
    }
  }
}

function move(element: HTMLElement, from: DOMRect, to: DOMRect, duration: number, fade: boolean): void {
  const dx = from.left + from.width / 2 - (to.left + to.width / 2);
  const dy = from.top + from.height / 2 - (to.top + to.height / 2);
  const scale = Math.max(0.3, Math.min(3, Math.max(from.width, from.height) / Math.max(to.width, to.height)));
  element.animate([
    { translate: `${dx}px ${dy}px`, scale: `${scale}`, opacity: fade ? 0.2 : 1 },
    { translate: '0 0', scale: '1', opacity: 1 },
  ], { duration, easing: EASE });
}

/** Whether the person asked for reduced motion; read once per frame. */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
