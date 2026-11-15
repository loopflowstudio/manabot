/** Same-origin art requests are driven only by names already visible to a viewer. */
export type ArtCredit = { artist: string; source: string; copyright: string };
const requests = new Map<string, { promise: Promise<boolean>; expires: number }>();
const shown = new Set<string>();

export function artUrl(name: string, kind: 'art' | 'full' = 'art'): string {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return `/api/card-art/${kind}/${slug}.jpg`;
}

export function loadArt(url: string, name?: string): Promise<boolean> {
  const existing = requests.get(url);
  if (existing && existing.expires > Date.now()) return existing.promise;
  const entry = { expires: Infinity, promise: Promise.resolve(false) };
  entry.promise = new Promise<boolean>(resolve => {
    const image = new Image();
    image.onload = () => {
      if (name && !shown.has(name)) {
        shown.add(name);
        window.dispatchEvent(new Event('academy-art-loaded'));
      }
      resolve(true);
    };
    image.onerror = () => { entry.expires = Date.now() + 600_000; resolve(false); };
    image.src = url;
  });
  requests.set(url, entry);
  return entry.promise;
}

export async function artCredits(): Promise<[string, ArtCredit][]> {
  if (!shown.size) return [];
  const params = new URLSearchParams();
  for (const name of shown) params.append('names', name);
  const response = await fetch(`/api/card-art/credits?${params}`);
  if (!response.ok) return [];
  return Object.entries(await response.json() as Record<string, ArtCredit>)
    .sort(([left], [right]) => left.localeCompare(right));
}
