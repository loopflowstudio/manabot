# T2 demo: TLA casting, costs, zone permissions (Task 87e012), 2026-10-29

Headless decision pass 1 through `managym.Env` (extension at `63e5fe407`), `skip_trivial` on, random policy, policy input v5. Academy was not driven in a browser.

Decks: a red/black/blue deck (Sozin's Comet, Fire Nation Attacks, Zuko, Iroh, Azula,
Hama, Barrels, Canyon Crawler, Zei, …) against a green/blue deck (Rockalanche, Toph,
Badgermole Cub, Earthbending Lesson, Giant Koi, Hermitic Herbalist, White Lotus
Hideout, Jasmine Dragon Tea Shop, Serpent of the Pass, Spirit Water Revival,
Solstice Revelations, …). Scripts: `scratch/tools/t2_demo*.py`; logs: `scratch/t2-demo/`.

## Observed

- `managym.unsupported_cards` over all 51 T2 cards: `[]` (`run.log`).
- 60 random games ended with winners 24/36 and no errors or panics (`run.log`).
- Restricted mana is offered with its class named: "Add {G} (Lesson or Shrine spells
  only)", "Add {R} (Ally spells or abilities of Ally sources only)", and the Herbalist
  two-colour Lesson pairs (`vocab.log`).
- Landcycling searches ("Reveal Forest and put it into your hand"), delayed
  earthbend returns ("Return this card to the battlefield tapped") and firebending
  attack triggers all appear (`vocab.log`).
- Zone casts were offered and taken over 60 games (`zones.log`):
  - Flashback from the graveyard: Fire Nation Attacks 113/44, Rockalanche 213/41,
    Solstice Revelations 174/34.
  - Zuko's exact-card impulse from exile: Barrels, Fire Nation Attacks, Zei and Zuko
    as casts, plus Island, Mountain and Swamp as land plays.
  - Foretold Sozin's Comet cast from exile: 202 offered, 60 taken.
- Foretell trace, seed 0 (`foretell.log`):
  - On turn 13 the owner sees Sozin's Comet in exile, face down, with a `foretold`
    grant and `after_turn 13`.
  - The opponent sees no exile card, only an exile count of 1. "Sozin" does not appear
    in the opponent's observation JSON.
  - The cast from exile was first offered on turn 15, a later turn.
  - The v5 buffers have player 60, card 55, permanent 32 and relations 33 columns.

## Judgment (interpretation, not a human statement)

T2 holds through the bot/training path: cards are supported, each non-hand
permission seen (flashback, exile impulse, foretell) is offered only where the rules
allow it, and foretell stays private. No defect; advanced to the gate.

## Open / next

- In-browser Academy proof (a person foretelling and casting from exile) is left to PR review (`./scripts/play`) or Playwright.
- Not in the demo decks, covered by directed tests in `managym/tests/cards/tla.rs`: Merchant of Many Hats, Yue, Avatar's Wrath, Lost Days.

## Re-runs (2026-10-29 – 2026-10-30, through `b678524c1`)

No `managym/src` or extension change since `63e5fe407`; T1 #394 still open at
`2c50c8793`. Each re-run of `t2_demo.py … 60` and `t2_demo_foretell.py` was
byte-identical to `run.log` and `foretell.log` (60 games, winners 24/36, no
errors, 0 face-down leaks). Advance.

2026-10-30 decision pass 1, at `b8dbfa319` (synced onto T1 `e6b672753`; #394 still
open): the implement pass in between made no tracked change. Since `63e5fe407`, the only
runtime change is `d130bbc90` in `text_observation.py`. Re-ran `t2_demo.py
scratch/t2-demo/t2-cards.json 60` and `t2_demo_foretell.py`, and both outputs matched
`run.log` and `foretell.log` byte for byte. Advance.
