/**
 * Keyboard regions of the table: Tab moves between regions, arrows within one.
 * A region with no controls (an empty field) is a stop of its own.
 *
 * `region` is a Svelte action on a region's element (the opponent's field, the
 * stack, your field, the hand, the decision, the legal choices, the moves).
 * It keeps one control of the region in the Tab order (a roving tabindex): the
 * one last focused there, else the first. Arrow keys move focus to the
 * nearest control in that direction on screen, so a field's two rows and a
 * fanned hand read as they look; where nothing lies that way (a list's
 * left/right), they step through the region in document order. On a
 * recording (`across: false`) Left and Right are the transport's, stepping
 * positions; Up and Down still move within the region.
 *
 * A move replaces the controls it was made from. When the focused control
 * leaves the page during keyboard play, focus returns to the control now at
 * its place in the same region, so a keyboard-only game never falls back to
 * the top of the page. Pointer play is untouched: nothing is focused for it.
 */

const CONTROLS = 'button:not([disabled]), a[href], summary, [tabindex]:not([data-region])';
const ACTS = '[data-offer-id], [data-offer-count]:not([data-offer-count="0"])';

/** The last input was a key, not a pointer: only then does focus follow a replaced control. */
let keyboard = false;
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', () => { keyboard = true; }, true);
  window.addEventListener('pointerdown', () => { keyboard = false; }, true);
}

function controls(node: HTMLElement): HTMLElement[] {
  return [...node.querySelectorAll<HTMLElement>(CONTROLS)].filter(element =>
    element.closest('[data-region]') === node && element.getClientRects().length > 0);
}

type Direction = 'left' | 'right' | 'up' | 'down';
const DIRECTIONS: Record<string, Direction> = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };

/** The control nearest `from` in `direction`, weighing distance across the direction twice. */
export function nearest(from: DOMRect, candidates: readonly DOMRect[], direction: Direction): number | null {
  const cx = from.left + from.width / 2;
  const cy = from.top + from.height / 2;
  let best: { index: number; score: number } | null = null;
  candidates.forEach((rect, index) => {
    const x = rect.left + rect.width / 2 - cx;
    const y = rect.top + rect.height / 2 - cy;
    const along = direction === 'left' ? -x : direction === 'right' ? x : direction === 'up' ? -y : y;
    const across = direction === 'left' || direction === 'right' ? Math.abs(y) : Math.abs(x);
    if (along <= 1) return;
    const score = along + 2 * across;
    if (!best || score < best.score) best = { index, score };
  });
  return (best as { index: number } | null)?.index ?? null;
}

export interface RegionOptions {
  /** Left and Right move within the region; false leaves them to the page (a recording's stepping). */
  across?: boolean;
}

export function region(node: HTMLElement, options: RegionOptions = {}): { update: (next: RegionOptions) => void; destroy: () => void } {
  let across = options.across ?? true;
  node.dataset.region = '';
  let current: HTMLElement | null = null;
  /** The focused control and its place in the region while focus is here; a removed control fires no focusout. */
  let held: { element: HTMLElement; index: number } | null = null;

  function rove(): void {
    const items = controls(node);
    if (current && !items.includes(current)) current = null;
    const active = items.find(item => item === document.activeElement);
    if (active) current = active;
    // Tab lands on the control last used here, else the first that acts (a card, a player or a choice with an offer).
    const entry = current ?? items.find(item => item.matches(ACTS)) ?? items[0];
    items.forEach(item => { item.tabIndex = item === entry ? 0 : -1; });
    // A region with nothing to press is still a stop, so its name and contents can be read.
    if (items.length) node.removeAttribute('tabindex');
    else node.tabIndex = 0;
  }

  function focus(item: HTMLElement): void {
    current = item;
    rove();
    item.focus();
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const items = controls(node);
    const at = items.indexOf(event.target as HTMLElement);
    if (at < 0) return;
    let next: number | null = null;
    if (event.key in DIRECTIONS) {
      const direction = DIRECTIONS[event.key];
      if (!across && (direction === 'left' || direction === 'right')) return;
      const rects = items.map(item => item.getBoundingClientRect());
      const others = rects.map((rect, i) => (i === at ? new DOMRect(-1e6, -1e6, 0, 0) : rect));
      next = nearest(rects[at], others, direction);
      if (next === null || next === at) next = direction === 'left' || direction === 'up' ? at - 1 : at + 1;
      if (next < 0 || next >= items.length) next = null;
    } else return;
    // A key inside a region moves within it; it does not also step a recording.
    event.preventDefault();
    event.stopPropagation();
    if (next !== null) focus(items[next]);
  }

  function onFocusin(event: FocusEvent): void {
    if (!(event.target instanceof HTMLElement) || event.target === node) return;
    current = event.target;
    rove();
    held = { element: current, index: controls(node).indexOf(current) };
  }

  function onFocusout(event: FocusEvent): void {
    // Focus moved somewhere else on purpose, or the page lost it to a pointer.
    if (event.relatedTarget !== null || !keyboard) held = null;
  }

  const observer = new MutationObserver(() => {
    const items = controls(node);
    const orphaned = document.activeElement === document.body || document.activeElement === null;
    // Only a control the page removed or disabled hands focus on; leaving the page by Tab does not.
    const gone = held && (!held.element.isConnected || held.element.matches(':disabled'));
    if (held && gone && keyboard && (orphaned || document.activeElement === held.element)) {
      if (items.length) { focus(items[Math.min(held.index, items.length - 1)]); return; }
      // Nothing left to press here (the choices while the opponent decides): the region keeps focus.
      rove();
      node.focus({ preventScroll: true });
      return;
    }
    // The region held focus while empty; its controls are back.
    if (document.activeElement === node && items.length) { focus(items.find(item => item.matches(ACTS)) ?? items[0]); return; }
    if (!orphaned && !node.contains(document.activeElement)) held = null;
    rove();
  });
  observer.observe(node, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled'] });
  node.addEventListener('keydown', onKeydown);
  node.addEventListener('focusin', onFocusin);
  node.addEventListener('focusout', onFocusout);
  rove();
  return {
    update(next: RegionOptions) { across = next.across ?? true; },
    destroy() {
      observer.disconnect();
      node.removeEventListener('keydown', onKeydown);
      node.removeEventListener('focusin', onFocusin);
      node.removeEventListener('focusout', onFocusout);
    },
  };
}
