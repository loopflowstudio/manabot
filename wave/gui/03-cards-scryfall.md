# Stage 3: Card rendering with Scryfall images

## What to build

Replace text-only card rendering with actual card images from Scryfall. Cards in hand, on the battlefield, in the graveyard — all show their art. Tapped permanents rotate 90 degrees. Hover shows full-size card.

## Key functions

```typescript
// Scryfall image URL by exact card name
function scryfallImageUrl(cardName: string, size: "small" | "normal" | "large" = "normal"): string {
  return `https://api.scryfall.com/cards/named?format=image&exact=${encodeURIComponent(cardName)}&version=${size}`
}

// Card component with image, tap state, selection highlight
function Card({ card, tapped, selected, onClick }: CardProps): JSX.Element

// Card back for hidden cards (opponent's hand, library)
function CardBack(): JSX.Element
```

## Constraints

- Use `small` size (146x204) for hand/graveyard, `normal` (488x680) for hover preview
- Browser caches images naturally — only 9 unique cards, so no custom caching needed
- Tapped permanents: CSS `transform: rotate(90deg)`
- Summoning-sick permanents: slight opacity reduction or visual indicator
- Damaged creatures: show damage counter overlay

## Done when

- All cards render with Scryfall images
- Tapped permanents appear rotated
- Hovering a card shows a larger preview
- Card backs shown for opponent's hand
- No broken images for any of the 9 implemented cards
