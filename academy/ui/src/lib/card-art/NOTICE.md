# Card art cache

Files in this directory are Magic: The Gathering card art crops and the
classic card back,
© Wizards of the Coast LLC, fetched once per machine from Scryfall's image
CDN by `academy/ui/scripts/fetch-card-art.mjs` (never at runtime, never
re-fetched once present). They are used as unofficial Fan Content permitted
under the Wizards of the Coast Fan Content Policy and are not committed to
the repository; when a file is absent the interface falls back to the
curated pack's procedural treatment. Not approved or endorsed by Wizards.

Run `npm --prefix academy/ui run fetch:card-art` from the repository root to
populate the cache for curated identities and checked demo decks. The generated
`credits.json` retains the artist and Scryfall source for each image; the app
shows them in its collapsed Card art credits section. Downloaded images remain
unfiltered; procedural grain and etching effects apply only to fallback art.
