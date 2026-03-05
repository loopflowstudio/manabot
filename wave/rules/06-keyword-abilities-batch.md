# 06: Keyword Abilities Batch

## Finish line

Evergreen keyword abilities are implemented as combat modifiers, dramatically
expanding strategic depth without requiring triggers, stack, or targeting
infrastructure.

## Changes

### Keywords (all flags + combat/SBA checks)

| Keyword | Implementation | CR |
|---------|---------------|----|
| Flying | Blocker legality: can't be blocked except by flying/reach | 702.9 |
| Reach | Blocker legality: can block creatures with flying | 702.17 |
| Haste | Suppress summoning sickness on ETB | 702.10 |
| Vigilance | Don't tap when declared as attacker | 702.20 |
| Trample | Excess combat damage assigned to defending player | 702.19 |
| First Strike | Create first combat damage sub-step | 702.7 |
| Double Strike | Assign damage in both first strike and normal sub-steps | 702.4 |
| Deathtouch | Any damage is lethal for SBA 704.5h | 702.2 |
| Lifelink | Damage dealt causes controller to gain that much life | 702.15 |
| Defender | Can't be declared as attacker | 702.3 |
| Menace | Must be blocked by 2+ creatures | 702.111 |

### Card Data Model

Add keyword flags to `Card` / `Permanent`:

```rust
pub struct Keywords {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
}
```

Keywords are part of the observation space immediately — the agent can see
them as features on permanents.

### Card Pool Expansion

Add cards to exercise keywords (suggested, not exhaustive):

- Serra Angel (3WW, 4/4, flying, vigilance)
- Shivan Dragon (4RR, 5/5, flying) — or simpler flyer
- Giant Spider (3G, 2/4, reach)
- Goblin Guide (R, 2/2, haste)
- Typhoid Rats (B, 1/1, deathtouch)
- Trained Armodon (1GG, 3/3, trample)
- Wall of Omens (1W, 0/4, defender) — ETB draw if triggers exist, otherwise vanilla defender

### First/Double Strike Combat

The main non-trivial implementation. Combat damage step splits into two
sub-steps when any creature has first strike or double strike:
1. First strike damage step: only first/double strikers assign damage
2. Normal damage step: only non-first-strikers and double strikers assign

SBAs checked between sub-steps (creatures can die to first strike before
dealing normal damage).

### Trace Tests

Per keyword:
- Positive: keyword functions correctly (flyer can't be blocked by ground creature)
- Negative: keyword absence works (non-flyer CAN be blocked by ground creature)
- Interaction: deathtouch + trample, first strike kills before normal damage

### Training smoke test

Compare agent behavior with/without keywords. Expect richer combat decisions
(trading, evasion, chump blocking).

## Done when

- All 11 keywords functional with CR-cited trace tests.
- Card pool expanded with keyword-bearing creatures.
- First/double strike combat damage sub-steps work correctly.
- Training smoke test shows changed combat patterns.
