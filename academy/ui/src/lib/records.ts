const KEY = 'etude.play.record-credentials';
const PLAYER_KEY = 'etude.play.player-credential';

export function playerToken(): string | undefined {
  if (typeof localStorage === 'undefined') return undefined;
  let token = localStorage.getItem(PLAYER_KEY);
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem(PLAYER_KEY, token);
  }
  return token;
}

function credentials(): string[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const value: unknown = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  } catch {
    return [];
  }
}

export function rememberRecordCredential(token: string): void {
  const tokens = credentials();
  if (!tokens.includes(token)) localStorage.setItem(KEY, JSON.stringify([...tokens, token]));
}

function watchCredentials(): string[] {
  if (typeof sessionStorage === 'undefined') return [];
  const tokens: string[] = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (key?.startsWith('etude.watch.')) {
      const token = sessionStorage.getItem(key);
      if (token) tokens.push(token);
    }
  }
  return tokens;
}

export function recordFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const player = playerToken();
  if (player) headers.set('x-etude-player-token', player);
  const tokens = [...credentials(), ...watchCredentials()];
  headers.set('x-etude-participant-tokens', tokens.join(','));
  if (tokens.length && !headers.has('x-etude-participant-token')) headers.set('x-etude-participant-token', tokens[tokens.length - 1]);
  return fetch(input, { ...init, headers });
}
