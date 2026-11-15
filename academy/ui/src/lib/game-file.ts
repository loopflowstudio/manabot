/** Split recorded games without rewriting their integer identities or digests.
 * JSON.parse validates shape only. The importer sends each original source span.
 */
export function recordedGames(source: string): string[] {
  const text = source.trim();
  let parsed: unknown;
  try { parsed = JSON.parse(text); }
  catch {
    const lines = text.split('\n').filter(line => line.trim());
    if (!lines.length) throw new Error('Expected a game record.');
    return lines.map(record);
  }
  if (!Array.isArray(parsed)) return [record(text)];
  const games: string[] = [];
  let start = 1;
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let i = 1; i < text.length - 1; i++) {
    const char = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '{' || char === '[') depth++;
    else if (char === '}' || char === ']') depth--;
    else if (char === ',' && depth === 0) { games.push(record(text.slice(start, i))); start = i + 1; }
  }
  if (parsed.length) games.push(record(text.slice(start, -1)));
  if (!games.length) throw new Error('No games in this file.');
  return games;
}
function record(text: string): string {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a JSON game object.');
  return text.trim();
}
