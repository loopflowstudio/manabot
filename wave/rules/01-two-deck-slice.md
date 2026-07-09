# 01: The Two-Deck Slice (Milestone 1)

Two 40-card decks from the elemental cube, playable against each other in the
engine and behind the play UI. Lists blessed 2026-07-09 ("don't care on the
specific 40 too much — I'll be impressed if we can get this set working"), so
counts may flex ±1 during implementation; the CAPABILITY BILL is the contract.

## UR Lessons (17 lands: 9 Island, 8 Mountain)

2 Tiger-Seal · 2 Otter-Penguin · 2 Fire Nation Cadets · 2 First-Time Flyer ·
1 Forecasting Fortune Teller · 1 Dragonfly Swarm · 4 Firebending Lesson ·
2 Igneous Inspiration · 2 Pop Quiz · 2 Divide by Zero · 2 It'll Quench Ya! ·
2 Accumulate Wisdom

Learn note: no sideboard in 1v1 constructed → learn = "you may discard, then
draw" (rules-correct without a wishboard).

## GW Allies (17 lands: 9 Plains, 8 Forest)

2 Water Tribe Rallier · 2 Invasion Reinforcements · 2 Compassionate Healer ·
2 Earth Kingdom Jailer · 2 White Lotus Reinforcements · 2 Earth King's
Lieutenant · 2 Kyoshi Warriors · 2 Badgermole Cub · 1 Suki, Kyoshi Warrior ·
1 South Pole Voyager · 2 Allies at Last · 1 Yip Yip! · 2 Fancy Footwork

(Badgermole Cub added by owner call — earthbend is IN Milestone 1.)

## The capability bill (the contract)

Staged for dispatch; each stage lands with cargo trace-tests per capability
and observation-encoding for anything an agent must be able to see.

**Stage 1 — trigger substrate & board state** (dispatched 2026-07-09):
triggered-ability framework generalized over sources (ETB, death, attack,
upkeep, becomes-tapped, taps-for-mana, another-[type]-enters,
Nth-card-drawn-this-turn, second-time-this-turn gating); **+1/+1 counters**
(state, P/T contribution, on lands too — earthbend prereq); **tokens**
(creature tokens incl. entering tapped-and-attacking; Clue artifact token
with sac-to-draw); **type predicates** (Ally/Lesson checks, graveyard type
counting); keywords: flash, vigilance-correctness, conditional unblockable.
Observation encoding: counter counts, token flag, type tags, gy-type counts.

**Stage 2 — decisions & costs:** mid-resolution player choices (scry,
look-N-select, pay-or-not [kicker / ward / It'll Quench Ya!], modal);
waterbend (tap-permanents-to-help-pay); affinity-style cost reduction;
multi-target spells; spell-bounce (Divide by Zero).

**Stage 3 — specials:** earthbend (land animation + counters + dies/exiled →
return tapped); exile-until-this-leaves linkage (Earth Kingdom Jailer);
static continuous effects (anthem — White Lotus Reinforcements); dynamic P/T
(Suki, Dragonfly Swarm); learn (discard-then-draw mode); until-end-of-combat
mana (firebending); death triggers with gy conditions.

**Stage 4 — registration & validation:** all 25 distinct nonland cards
registered in a `tla` cardset module; per-card trace tests; ≥200
random-vs-random A-vs-B games to terminal with zero errors; decks exposed to
the play UI and the training/eval stack (deck constants next to
INTERACTIVE_DECK); first A-vs-B matchup table (seat-balanced, per-deck).

## Notes

- Cub + Rallier interaction is a real test case: waterbend taps creatures to
  pay; Cub's trigger adds {G} per creature tapped for mana — the trigger and
  the payment mechanic must compose.
- Milestone 2 (TLA commons complete, `00-pool-audit.md`) reuses every rung
  here; nothing in this slice is throwaway.
