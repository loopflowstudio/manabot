/** This tab's last game destination, independent of its execution credentials. */
export const currentGame = $state<{href: string | null}>({href:null});
export function rememberCurrentGame(href: string): void {
  if (href !== '/' && !/^\/replay\?trace=[^&]+$/.test(href)) return;
  currentGame.href = href;
  try { sessionStorage.setItem('academy.current-game', href); } catch { /* Navigation remains usable in memory. */ }
}
export function loadCurrentGame(): void {
  try {
    const href = sessionStorage.getItem('academy.current-game');
    if (href) rememberCurrentGame(href);
    else if (sessionStorage.getItem('etude.gui.resume')) currentGame.href = '/';
  } catch { /* There is no stored destination. */ }
}
