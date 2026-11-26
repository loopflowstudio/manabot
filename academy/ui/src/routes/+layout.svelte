<script lang="ts">
  import { onMount, type Snippet } from 'svelte';

  import { page } from '$app/state';
  import ThemeSwitch from '$lib/components/ThemeSwitch.svelte';
  import { followTheme } from '$lib/theme.svelte';

  import '../app.css';

  let { children }: { children?: Snippet } = $props();

  onMount(followTheme);

  const NAV = [
    { href: '/', label: 'Play' },
    { href: '/games', label: 'History' },
  ];
  const isActive = (href: string): boolean => (href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href));
</script>

<svelte:head>
  <title>Academy</title>
</svelte:head>

<div class="flex min-h-screen flex-col text-ink">
  <!-- One bar on every page: paper, with the color pie as a thin W-U-B-R-G rule beneath it. -->
  <header class="chrome-bar">
    <div class="mx-auto flex w-full max-w-[1600px] items-center justify-between px-4 py-2">
      <div data-testid="brand-name" class="type-brand text-display">Academy</div>
      <nav class="flex items-center gap-2">
        {#each NAV as item (item.href)}
          <a
            href={item.href}
            aria-current={isActive(item.href) ? 'page' : undefined}
            class={`rounded border px-3 py-1 transition ${
              isActive(item.href)
                ? 'border-line bg-field font-semibold text-ink'
                : 'border-transparent text-ink-2 hover:bg-field hover:text-ink'
            }`}
          >
            {item.label}
          </a>
        {/each}
        <span class="text-ink-2 hover:text-ink"><ThemeSwitch /></span>
      </nav>
    </div>
  </header>

  <div class="flex-1">
    {@render children?.()}
  </div>
</div>
