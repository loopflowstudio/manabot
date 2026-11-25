<script lang="ts">
  /** The light/dark switch: one press changes the theme and remembers it. */
  import { setTheme, theme } from '$lib/theme.svelte';

  const dark = $derived(theme.current === 'dark');
</script>

<button type="button" class="theme-switch type-caption" role="switch" aria-checked={dark} aria-label="Dark theme"
  title={dark ? 'Dark theme · press for light' : 'Light theme · press for dark'} data-testid="theme-switch"
  onclick={() => setTheme(dark ? 'light' : 'dark')}>
  <span class="track" aria-hidden="true"><span class="thumb"></span></span>
  <span>{dark ? 'Dark' : 'Light'}</span>
</button>

<style>
  .theme-switch { display: inline-flex; align-items: center; gap: 6px; padding: 2px 6px; border-radius: var(--radius-sm); color: inherit; white-space: nowrap; }
  .track { width: 26px; height: 14px; padding: 1px; border-radius: 7px; border: 1px solid currentColor; display: inline-flex; opacity: 0.8; }
  .thumb { width: 10px; height: 10px; border-radius: 5px; background: currentColor; transition: transform var(--motion-fast) ease-out; }
  [aria-checked='true'] .thumb { transform: translateX(12px); }
  .theme-switch:hover .track { opacity: 1; }
  @media (prefers-reduced-motion: reduce) { .thumb { transition: none; } }
</style>
