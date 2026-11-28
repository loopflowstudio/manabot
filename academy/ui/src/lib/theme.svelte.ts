/**
 * The light or dark theme. `app.html` sets `data-theme` on the root element
 * before first paint; this module reads it, follows the system's preference
 * until the reader chooses, and keeps a choice in `etude.theme` across pages
 * and tabs.
 */

export type Theme = 'light' | 'dark';

const KEY = 'etude.theme';

export const theme = $state<{ current: Theme }>({ current: 'light' });

function stored(): Theme | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === 'light' || value === 'dark' ? value : null;
  } catch {
    return null;
  }
}

function apply(next: Theme): void {
  theme.current = next;
  document.documentElement.dataset.theme = next;
}

/** Choose a theme and remember it. */
export function setTheme(next: Theme): void {
  apply(next);
  try { localStorage.setItem(KEY, next); } catch { /* The choice lasts for this page. */ }
}

/** Follow the root element's theme, the system's until a choice is stored, and other tabs. Returns the cleanup. */
export function followTheme(): () => void {
  const system = matchMedia('(prefers-color-scheme: dark)');
  apply(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  const onSystem = (): void => { if (stored() === null) apply(system.matches ? 'dark' : 'light'); };
  const onStorage = (event: StorageEvent): void => {
    if (event.key === KEY) apply(stored() ?? (system.matches ? 'dark' : 'light'));
  };
  system.addEventListener('change', onSystem);
  window.addEventListener('storage', onStorage);
  return () => {
    system.removeEventListener('change', onSystem);
    window.removeEventListener('storage', onStorage);
  };
}
