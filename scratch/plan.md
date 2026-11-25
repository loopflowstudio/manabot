# MTG-432 · Foundations characteristics, replacements and combat constraints

All 29 FDN cohort rows (characteristics, replacement_events, combat_constraints) were
`definition pending` at base `cf73d9778f`; each slice built the shared primitive it
needed here. Oracle text is Scryfall's; set files are first printings, promos included.
Main's #593 (`45ddf3bca7`, synced `b3485ac26c`) supplies `CreatureTypes::All`.

MTG-431 overlap: Three Tree Mascot needs changeling (slice 4, `e8f5bd548a`) and Dragon
Mage needs whole-hand discard (`Effect::DiscardHand`, slice 6, `b2da13add6`). Hand
MTG-431 those revisions and paths (`questions.md`).

## Built

Cards: Confiscate, Harmless Offering, Joraga Invocation, Vampire Soulcaller, Knight of
Grace, Knight of Malice, Elenda, Sphinx of the Final Word, Progenitus, Crystal Barricade,
Taurean Mauler, Tempest Djinn, Blanchwood Armor, Dreadwing Scavenger, Dwynen, Garruk's
Uprising, Crossway Troublemakers, New Horizons, Darksteel Colossus, Dryad Militant,
Wilt-Leaf Liege, Myojin of Night's Reach, Mossborn Hydra, Heroes' Bane, Wildwood Scourge,
Prime Speaker Zegana, Giada, Banner of Kinship, Angel of Vitality.

1. CR 613.1b layer-2 control (`40ba5de05a`, `cace2b71ee`).
2. Block offers maximize satisfied requirements over the whole declaration (`6b9f01345d`).
3. Source quality (hexproof/protection, with LKI), player hexproof (same commits).
4. Changeling and counted bonuses (`e8f5bd548a`).
5. `ReplacementEffect::GraveyardInstead` from every origin; milled Colossus reveals and
   shuffles in place with a fresh identity, CR 701.20d (`dac0dd6d57`).
6–8. Whole-hand discard batch, shared entry-counter replacement, Angel of Vitality and
   Barricade prevention (`b2da13add6`; Tinybones `ForPlayer` fix `a4504cdc6b`).
- Pass 6 integrated positions (six directed multi-card positions, FDN ledger updated).
- Pass 7 conceptual alignment (`9fcab2395c`): `entry_controller_id`, `SourceQuality`
  wire names, `CombatState.blocked` (CR 509.1h, 506.4), `Condition::CastFrom`.
- Decomposition D1–D3 (`1eec144255`, `384e717f6a`, `1eedafb796`): one layer-7 quantity
  (`LayerValue::Count`), one entry quantity (`EntersWithCounters.count: EffectValue`), one
  damage replacement (`DamageSource`/`DamageTransform`); stale consumers fixed
  (`338f79de16`). Review at `d1887e600c`: composition holds, no card-keyed engine code.
- Performance P1–P4 (`8fa265d0a8`): layer dirty bits, inlined subtype checks, direct
  control scan, `Operation<i32>` resolution and bitmask `TargetingSource`.
- Performance P5–P7: single baseline matching + residual reachability for non-menace
  `block_options`, allocation-free `Color::bits` comparison in `Observation::add_permanent`,
  and `active_layers: u16` bitmask skipping + reusable buffers + single-batch exit +
  direct granted-static node emission in `Context::evaluate`.

Findings that still explain the code:

- Menace after ordinary matching can reject a legal tied completion. The solver reserves
  a partner for a committed singleton and branches on ambiguous menace attackers; work
  grows exponentially in the number of ambiguous menace attackers. Non-menace coupled
  declarations solve one baseline transversal-matroid matching and evaluate all `A + 1`
  choices via one descending-weight residual reachability pass (`O(A log A + E)`).
- Search determinization must bind private discard payments to matching hand cards
  (`selection_sampling.rs`); whole-hand batches preserve committed hand slots.
- Retiring newtype enum variants needs both the variant and its closing `/Enum::Variant`
  structure token in `token_renames.rs` for pinned pilots to replay unchanged.

### Performance (Apple M4 Max, uninstrumented release, digests identical to `baseline.json`)

Against base `3be5a0cfb2` after P1–P7 (`1f45d77cca`): `bench_steps` instructions `46.737B →
45.194B` (-3.30%); `bench_search` instructions `1,027.450B → 959.523B` (-6.61%), cycles -8.02%,
wall `59.990 s → 54.161 s` (-9.72%). Host was contended (load ~17); wall numbers are indicative.
`joraga-blocks` size 32: actions -52/-69%, search `47.4 → 19.6 ms`. Debug suites (638 engine
tests, 23 card suites, `bench_states` units) passed at `9b73df7caa`; P5–P7 reran targeted
debug tests (block solver exhaustive 3×3 cross-check, 41 combat/layer rules tests, FDN cards).

Ruled out or outside this Task: `refresh_control` self cost <0.6% (no bookkeeping flags);
`Arc<Characteristics>::make_mut` (~7% search) is MTG-420 architecture; `myojin-discard` and
`quality-targets` growth is inherent O(N²) rules work.

## Remaining

1. Gate: full debug nextest suite (CI profile, `CARGO_BUILD_JOBS=2`, `--test-threads 2`),
   Python text tests on a rebuilt extension, and Engine Budget on `ubuntu-24.04`:
   `gh workflow run ci.yml --ref jack-heart/df83ea-close-foundations-characteristic -f measure=<commit>`
   (record a new reference only with its reason). No x86-64 wall-time claim before then.
