<script lang="ts">
  import type { Snippet } from 'svelte';

  import { page } from '$app/state';

  import '../app.css';

  let { children }: { children?: Snippet } = $props();

  const NAV = [
    { href: '/', label: 'Play' },
    { href: '/replay', label: 'Replay' },
  ];

  function isActive(href: string): boolean {
    return href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href);
  }
</script>

<svelte:head>
  <title>Etude Fantasia</title>
</svelte:head>

<div class="flex min-h-screen flex-col text-ink">
  <!-- The banner carries the color pie as a W→U→R→B→G weave: each color
       radiates from its own point, alternating edges, in the manner of the
       loopflow logo's multi-direction gradient. Like the card name plates,
       the banner is a fixed rich world in both modes — Lotus Cobra
       saturation under a title scrim, literal ivory text. -->
  <header
    class="border-b border-black/30"
    style="background:
        linear-gradient(180deg, rgb(15 11 6 / 0.34), rgb(15 11 6 / 0.2)),
        radial-gradient(ellipse 42% 170% at 2% 0%, color-mix(in srgb, var(--vivid-w) 85%, transparent), transparent 78%),
        radial-gradient(ellipse 42% 170% at 26% 100%, color-mix(in srgb, var(--vivid-u) 85%, transparent), transparent 78%),
        radial-gradient(ellipse 42% 170% at 50% 0%, color-mix(in srgb, var(--vivid-r) 85%, transparent), transparent 78%),
        radial-gradient(ellipse 42% 170% at 74% 100%, color-mix(in srgb, var(--vivid-b) 85%, transparent), transparent 78%),
        radial-gradient(ellipse 42% 170% at 98% 0%, color-mix(in srgb, var(--vivid-g) 85%, transparent), transparent 78%),
      #221a10"
  >
    <div class="mx-auto flex w-full max-w-[1600px] items-center justify-between px-4 py-3">
      <div class="flex items-baseline gap-3">
        <div data-testid="brand-name" class="type-brand text-[#f8f1e0]">Etude Fantasia</div>
        <div class="type-rubric hidden text-[#f8f1e0]/65 sm:block">Play · replay · study</div>
      </div>
      <nav class="flex items-center gap-2">
        {#each NAV as item}
          <a
            href={item.href}
            aria-current={isActive(item.href) ? 'page' : undefined}
            class={`rounded border px-3 py-2 transition ${
              isActive(item.href)
                ? 'border-[#f8f1e0]/30 bg-[#f8f1e0]/15 font-semibold text-[#f8f1e0]'
                : 'border-transparent text-[#f8f1e0]/80 hover:border-[#f8f1e0]/25 hover:bg-[#f8f1e0]/10 hover:text-[#f8f1e0]'
            }`}
          >
            {item.label}
          </a>
        {/each}
      </nav>
    </div>
  </header>

  <div class="flex-1">
    {@render children?.()}
  </div>

  <!-- The colophon: the book closes with its imprint — identity, ways in,
       and the attribution the Fan Content Policy asks us to show. -->
  <footer class="mt-10 border-t border-line">
    <div class="mx-auto grid w-full max-w-[1400px] gap-x-12 gap-y-6 px-4 py-8 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,22rem)]">
      <div>
        <div class="type-title text-display">Etude Fantasia</div>
        <p class="type-caption mt-1 max-w-[48ch] text-ink-2">
          An étude in studying Magic: play a manabot, replay every decision,
          and annotate the score.
        </p>
      </div>
      <nav aria-label="Footer" class="type-caption">
        <div class="type-rubric text-ink-2">Table</div>
        <ul class="mt-1">
          {#each NAV as item}
            <li><a class="inline-block py-1 text-ink-2 underline-offset-2 hover:text-ink hover:underline" href={item.href}>{item.label}</a></li>
          {/each}
        </ul>
      </nav>
      <p class="type-caption text-ink-3">
        Etude Fantasia is unofficial Fan Content permitted under the
        <a
          class="underline underline-offset-2 hover:text-ink"
          href="https://company.wizards.com/en/legal/fancontentpolicy"
          rel="external noopener">Wizards of the Coast Fan Content Policy</a
        >. Card art and mana symbols are © Wizards of the Coast. Not
        approved or endorsed by Wizards.
      </p>
    </div>
  </footer>
</div>
