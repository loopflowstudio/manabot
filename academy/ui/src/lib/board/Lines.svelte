<script lang="ts">
  /**
   * Lines over the board that join a card to what it acts on: a spell on the
   * stack to each target, a blocker to the attacker it blocks. Lines are drawn
   * from standing facts (`boardLines`), measured from where the endpoints are
   * drawn, and never take pointer input. Each line is also said in words: the
   * stack entry, the blocker's accessible name and the log carry the same facts.
   */
  import type { Endpoint, BoardLine } from './recent';

  interface Props {
    lines: BoardLine[];
    root: HTMLElement | null;
    /** Changes whenever the board may have moved underneath the lines. */
    revision: number;
  }

  let { lines, root, revision }: Props = $props();

  type Drawn = { key: string; kind: BoardLine['kind']; d: string; x: number; y: number };
  let drawn = $state<Drawn[]>([]);

  function element(endpoint: Endpoint): HTMLElement | null {
    if (!root) return null;
    const selector = endpoint.kind === 'player' ? `[data-testid=player-plate][data-player-index="${endpoint.index}"] [data-testid=player-target]`
      : endpoint.kind === 'stack' ? `[data-stack-card="${endpoint.id}"]`
        : `.board [data-testid=board-card][data-object-id="${endpoint.id}"]`;
    return [...root.querySelectorAll<HTMLElement>(selector)].find(candidate => candidate.getBoundingClientRect().width > 0) ?? null;
  }

  function measure(): void {
    drawn = lines.flatMap(line => {
      const from = element(line.from)?.getBoundingClientRect();
      const to = element(line.to)?.getBoundingClientRect();
      if (!from || !to) return [];
      // A stack entry is a row of text in the rail; start at its card, not across its name.
      const a = { x: line.from.kind === 'stack' ? from.left + 14 : from.left + from.width / 2, y: from.top + from.height / 2 };
      const b = { x: to.left + to.width / 2, y: to.top + to.height / 2 };
      // Stop at the target's edge, so the arrow points at the card rather than under it.
      const inset = Math.min(to.width, to.height) / 2 + 4;
      const length = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      const end = { x: b.x - ((b.x - a.x) / length) * inset, y: b.y - ((b.y - a.y) / length) * inset };
      const bend = Math.min(80, length * 0.18);
      const mid = { x: (a.x + end.x) / 2 - ((end.y - a.y) / length) * bend, y: (a.y + end.y) / 2 + ((end.x - a.x) / length) * bend };
      return [{ key: JSON.stringify(line), kind: line.kind, d: `M ${a.x} ${a.y} Q ${mid.x} ${mid.y} ${end.x} ${end.y}`, x: end.x, y: end.y }];
    });
  }

  /** Longest card travel after a frame (a cast flying from the hand to the stack), in ms. */
  const TRAVEL_MS = 400;

  // Remeasure every frame while cards travel, so a line follows its endpoints
  // instead of pointing from where a card was, then on any resize.
  $effect(() => {
    void revision;
    void lines;
    if (!root) return;
    const start = performance.now();
    let frame = 0;
    const follow = (now: number): void => {
      measure();
      if (now - start < TRAVEL_MS) frame = requestAnimationFrame(follow);
    };
    frame = requestAnimationFrame(follow);
    const observer = new ResizeObserver(() => measure());
    observer.observe(root);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); };
  });
</script>

<svelte:window onresize={measure} />

{#if drawn.length}
  <svg class="lines" aria-hidden="true" data-testid="board-lines">
    <defs>
      <marker id="line-target" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" class="head target" />
      </marker>
      <marker id="line-block" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6">
        <circle cx="5" cy="5" r="4" class="head block" />
      </marker>
    </defs>
    {#each drawn as line (line.key)}
      <path d={line.d} class={`line ${line.kind}`} pathLength="1" data-line-kind={line.kind}
        marker-end={`url(#line-${line.kind})`} />
    {/each}
  </svg>
{/if}

<style>
  .lines { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: var(--layer-lines); pointer-events: none; overflow: visible; }
  .line { fill: none; stroke-width: 2.25; stroke-linecap: round; stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw calc(var(--motion-move) * 2) var(--motion-move-ease); }
  .line.target { stroke: var(--accent); }
  .line.block { stroke: var(--text-secondary); stroke-dasharray: 0.02 0.018; animation: none; }
  .head.target { fill: var(--accent); }
  .head.block { fill: var(--text-secondary); }
  @keyframes draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
  @media (prefers-reduced-motion: reduce) {
    .line { animation: none; }
  }
</style>
