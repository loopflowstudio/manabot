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

### Performance review at `855d052a83` (Apple M4 Max, uninstrumented release, digests identical)

Rebuilt HEAD `855d052a83` (post-`bfd2cb42b5` compress) and base `3be5a0cfb2` on the same
contended host (load 33–63) without `--features benchmark`. All identities and digests
match base `3be5a0cfb2` and `managym/perf/baseline.json` bit-for-bit (`bench_steps`:
512 games, 118,559 steps, `10601622128491155944`; `bench_steps --layers`: 16 games, 1,308
steps, `1064912348938184597`; `bench_search` seeds 1–3 digests, `score_digest_blake3`,
`action_counts`, `simulations` 7680/8064/7424, `cap_hits` 0 identical). Because host load
varied widely during wall-clock runs, decisions rely on instructions retired, cycles, user
CPU, and 1 ms `sample` CPU profiles; x86-64 wall-time claims wait for CI Engine Budget.

| Workload | Base `3be5a0cfb2` | Pre-compress `1f45d77cca` | HEAD `855d052a83` | HEAD vs Base |
|---|---:|---:|---:|---:|
| `bench_steps` instructions (C/R/R/C avg) | 46.909B (46.877B / 46.940B) | 45.194B (load ~17) | 45.300B (45.220B / 45.381B) | -3.43% |
| `bench_steps` cycles (C/R/R/C avg) | 13.435B | — | 12.969B | -3.47% |
| `bench_steps --layers` instructions / cycles | 654.387M / 237.273M | — | 629.096M / 212.304M | -3.86% / -10.52% |
| `bench_search` instructions | 1,033.843B | 959.523B (load ~17) | 963.314B | -6.82% |
| `bench_search` cycles / user CPU | 329.594B / 143.82 s | — | 276.881B / 126.09 s | -15.99% / -12.33% |

- **Supported gains** (1 ms `sample` CPU profiles, scaled by instructions retired, HEAD vs Base):
  - `Context::evaluate`: `43.09% → 28.72%` (**-14.36%**) search, `23.72% → 22.14%` (**-1.58%**) steps; `Context::finish_object`: `8.66% → 1.21%` (**-7.45%**) search, `3.58% → 0.32%` (**-3.26%**) steps; `CreatureTypes::among`: `9.34% → 2.46%` (**-6.88%**) search, `4.76% → 0.80%` (**-3.96%**) steps.
  - `Game::layer_board` + `Game::refresh_control` combined: `41.57% → 33.72%` (**-7.85%**) search (`DerivedCache::invalidate` / `Arc<LayerBoard>::drop_slow` moved from `layer_board` into `refresh_control`), `23.96% → 23.72%` (**-0.24%**) steps.
  - `PrintedCharacteristics::colors`: `0.00%` steps / `0.07%` search (eliminating `9b73df7caa`'s `1.21%` steps regression via P6 `Color::bits`).
  - `Game::advance_event_batch` (including `collect_replacements`): `20.61% → 19.99%` (**-0.62%**) search; `combat_actions + block_options`: `4.07% → 3.99%` search, `1.02% → 0.93%` steps.
  - `bfd2cb42b5` (`lf compress`, direct granted-static emission and `BlockRequirements` helpers) preserved P5–P7 gains with zero regression (`bench_steps` -3.43%, `bench_search` -6.82% vs Base).
- **Large-board shapes (`bench_states --features benchmark`)**: all 9 touched shapes
  (`joraga-blocks`, `control-auras`, `counted-buffs`, `banner-choice`, `giada-entry`,
  `militant-mill`, `angel-life-gain`, `myojin-discard`, `quality-targets`) recorded zero
  dependency-clone `comparisons` and linear `boards`/`allocated_bytes` growth through size
  16. `joraga-blocks` through size 32 kept `block-0 actions` at `0.011 / 0.042 / 0.236 ms`
  and linear search `boards` (`88 / 168 / 328`).
- **Debug invariants at `855d052a83`** (all debug assertions active, including `block_requirements`
  residual cross-check, `layers` `active_layers` / `all_applied` / `finish_object`, and
  `layer_board` stale checks): 2,041 debug tests passed, 0 failed (`managym-state --lib` 18/18,
  `managym` lib + `rules_tests` + `scenario_tests` + `conformance_tests` 621/621 with 2
  skipped, all `managym-cards --test cards` 1,402/1,402).
- **Negative results, ruled out, and open bottlenecks outside MTG-432:**
  - `Arc<Characteristics>::make_mut` + `Arc<LayerObject>::make_mut` + counter `clone_from_ref`
    (~8.8% search: `Context::apply` 7.36%, counter loop 1.39%) is pre-existing MTG-420
    copy-on-write architecture — open engine-wide bottleneck outside this Task.
  - `Context::instructions` (~5.6% search) and `Game::layer_board` self / `compute_board`
    (~5.5% search) are pre-existing CR 613 static-scope and `BTreeMap` board-projection costs.
  - `Game::refresh_control` self (~1.3% search, <0.5% steps; remainder is moved `LayerBoard`
    drop): ruled out adding state-mutation bookkeeping flags across every write.
  - `myojin-discard` (`N` lords × `N` creatures), `quality-targets` (`N` Knights × `N`
    permanents), and `joraga-blocks` bipartite edge construction (`A × B`) are inherent O(N²)
    rules work. No credible profile-backed gain or unprofiled regression remains in MTG-432.

## Remaining

1. Gate: full debug nextest suite (CI profile, `CARGO_BUILD_JOBS=2`, `--test-threads 2`),
   Python text tests on a rebuilt extension, and Engine Budget on `ubuntu-24.04`:
   `gh workflow run ci.yml --ref jack-heart/df83ea-close-foundations-characteristic -f measure=<commit>`
   (record a new reference only with its reason). No x86-64 wall-time claim before then.
