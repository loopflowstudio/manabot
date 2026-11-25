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
  length-based `discover` filtering in `Context::evaluate`.

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

### Performance review at `9b73df7caa` (Apple M4 Max, uninstrumented release, digests identical)

Rebuilt candidate `9b73df7caa` (post-compress `c4bd9389a7`) and base `3be5a0cfb2` (same
engine as `managym/perf/baseline.json` reference `7d3ac1eb1c`) on the same contended host
(load ~17) without `--features benchmark`. All digests match `baseline.json` identically
(`bench_steps` 512 games, 118,559 steps, `10601622128491155944`; `bench_search` seeds 1–3
digests, `score_digest_blake3`, `action_counts`, and simulation counts identical).

| Workload | Base `3be5a0cfb2` | Pre-P1 `055e67dda6` | Pre-compress `8fa265d0a8` | HEAD `9b73df7caa` | HEAD vs Base |
|---|---:|---:|---:|---:|---:|
| `bench_steps` instructions (C/R/R/C med) | 46.737B | 49.395B | 45.990B | 46.205B | -1.14% |
| `bench_steps` cycles (median) | 10.958B | 11.710B | 10.757B | 10.726B | -2.12% |
| `bench_steps` wall / throughput (median) | 2.746 s (43,168/s) | 2.944 s (40,277/s) | 2.714 s (43,685/s) | 2.697 s (43,956/s) | +1.83%/s |
| `bench_search` instructions (C/R) | 1,027.450B | 1,105.336B | 981.060B | 983.443B | -4.28% |
| `bench_search` cycles (C/R) | 233.776B | 256.941B | 221.497B | 221.300B | -5.34% |
| `bench_search` wall (total / worst) | 59.990 s / 1.420 s | — | 55.722 s / 1.339 s | 55.375 s / 1.336 s | -7.69% / -5.92% |

Debug checks at `9b73df7caa` (post-compress, with `finish_object` and `layer_board` debug
assertions active): `rules_tests`, `scenario_tests`, `conformance_tests`, `managym --lib`,
`managym-state --lib` (638 tests), 23 card/observability/definition suites, and
`bench_states` unit tests (2/2) all passed.

- **Supported gains** (1 ms `sample` profiles, scaled to instructions retired, HEAD vs Base):
  - `Context::finish_object`: `7.39% → 0.48%` (**-6.94%**) search, `4.39% → 0.47%` (**-3.92%**) steps; `CreatureTypes::among`: `5.44% → 0.37%` (**-5.08%**) search.
  - `layer_board` / `Context::evaluate`: `-6.48%` / `-7.08%` search, `-3.83%` / `-3.81%` steps; `Node::resolved` (`0.86% → 0.00%`) eliminated.
  - `advance_event_batch` (including extracted `collect_replacements`): `-1.53%` search, `-1.89%` steps; `combat_actions + block_requirements` combined: `-0.31%` search, `-0.30%` steps.
  - `bench_states` (`--features benchmark`): `control-auras`, `counted-buffs`, `banner-choice`, `giada-entry`, `militant-mill`, and `angel-life-gain` scale linearly through size 16–32 (`myojin-discard` resolve `0.165 ms` at size 16, down 3x from pre-P1).
- **Negative results & regressions:**
  - `c4bd9389a7` (`lf compress`) added `+0.47%` instructions on `bench_steps` (`45.990B → 46.205B`) and `+0.24%` on `bench_search` (`981.060B → 983.443B`): `PrintedCharacteristics::colors` jumped `0.26% → 1.21%` of `bench_steps`, with `1.16%` called from `Observation::add_permanent` (`characteristics.colors != object.colors()`) allocating a `BTreeSet<Color>` per permanent.
  - `joraga-blocks` shows superlinear `bench_states` growth with zero menace creatures: `block-0 / actions` `0.001 / 0.005 / 0.022 / 0.120 ms` (sz 4/8/16/32; `16.1x` 4→16, `5.6x` 16→32), `block-1 / actions` `0.001 / 0.005 / 0.026 / 0.170 ms` (`20.0x` 4→16, `6.6x` 16→32), `block-0 / step:block` `0.010 / 0.017 / 0.046 / 0.211 ms` (`4.6x` 16→32), and `block-0 / search` `0.380 / 0.522 / 4.060 / 47.400 ms` (`11.7x` 16→32; `block-8 / search` `20.5x` 16→32) due to `A + 1` full Kuhn matchings from scratch per `block_options` call (`O(A^2 * E)` per blocker, `O(N^4)` over combat).
- **Open bottlenecks & ruled-out items:**
  - `Context::evaluate` empty-layer scans (`Vec<usize>` `pending` `1.84%` + filter closure `1.42%` + `evaluate` self `5.51%`, up `+0.45%` after adding `Layer::Control` to `Layer::ALL`) scan all `nodes` on every empty layer (target for P7).
  - `Game::refresh_control` net self cost is `0.62%` search / `0.00%` steps (`2.14%` / `1.05%` is `DerivedCache::invalidate` moved from `layer_board`); ruled out adding state-mutation bookkeeping flags for <0.6%.
  - `Arc<Characteristics>::make_mut` (`7.06%` search, `3.16%` steps vs base `6.56%` / `4.03%`) is pre-existing MTG-420 architecture — open engine-wide bottleneck outside MTG-432.
  - `myojin-discard` (`characteristics` `0.005 / 0.012 / 0.039 ms`, `7.9x` 4→16) and `quality-targets` (`characteristics` `0.003 / 0.006 / 0.012 / 0.034 ms`, `2.8x` 16→32) reflect inherent `O(N^2)` rules work (`N` lords × `N` creatures; `N` Knights checking `N` permanents).

### Performance pass P5–P7 (Apple M4 Max, uninstrumented release, digests identical)

Digests remain identical to `managym/perf/baseline.json` (`bench_steps` `10601622128491155944`; `bench_search` seeds 1–3 digests and `score_digest_blake3` unchanged).

- `bench_steps` instructions retired (3-run median): `45.194B` (`-2.19%` vs `9b73df7caa` `46.205B`, `-3.30%` vs base `3be5a0cfb2` `46.737B`).
- `bench_search` instructions retired / cycles / wall: `959.523B` instructions (`-2.43%` vs `9b73df7caa` `983.443B`, `-6.61%` vs base `3be5a0cfb2` `1,027.450B`), `215.023B` cycles (`-2.84%` vs `9b73df7caa`, `-8.02%` vs base), `54.161 s` total / `1.315 s` worst (`-9.72%` / `-7.39%` vs base).
- `bench_states --features benchmark joraga-blocks --sizes 4,8,16,32`:
  - `block-0 / actions`: `0.0028 / 0.0064 / 0.0183 / 0.0580 ms` (sz 32 `-51.7%` vs `0.120 ms`; `16→32` `5.6x → 3.16x`).
  - `block-1 / actions`: `0.0018 / 0.0055 / 0.0141 / 0.0521 ms` (sz 32 `-69.4%` vs `0.170 ms`; `16→32` `6.6x → 3.69x`).
  - `block-0 / step:block`: `0.0193 / 0.0273 / 0.0406 / 0.0932 ms` (sz 32 `-55.8%` vs `0.211 ms`; `16→32` `4.6x → 2.30x`).
  - `block-0 / search`: `0.6816 / 0.7017 / 2.9633 / 19.6450 ms` (sz 32 `-58.6%` vs `47.400 ms`; `16→32` `11.7x → 6.63x`; `block-8 / search` `20.5x → 13.67x`).
- Debug verification: `block_requirements` unit tests (including exhaustive 3-attacker/3-blocker non-menace residual solver vs general bound), 41 CR 613 / block / combat / structured-offer `rules_tests`, 9 FDN card tests (`joraga`, `vampire_soulcaller`, `confiscate`), and `bench_states` unit tests (2/2) all passed with debug cross-checks and `active_layers` / `finish_object` assertions enabled.

## Remaining

1. Gate: full debug nextest suite (CI profile, `CARGO_BUILD_JOBS=2`, `--test-threads 2`),
   Python text tests on a rebuilt extension, and Engine Budget on `ubuntu-24.04`:
   `gh workflow run ci.yml --ref jack-heart/df83ea-close-foundations-characteristic -f measure=<commit>`
   (record a new reference only with its reason). No x86-64 wall-time claim before then.
