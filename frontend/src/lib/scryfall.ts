export function scryfallImageUrl(
  name: string,
  size: 'small' | 'normal' = 'small',
): string {
  return `https://api.scryfall.com/cards/named?format=image&exact=${encodeURIComponent(name)}&version=${size}`;
}
