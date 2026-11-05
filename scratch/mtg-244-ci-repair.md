# MTG-244 CI repair — 2026-11-04/05

Jack Heart directed repair of PR #427 before review readiness, with no baseline
adoption, fewer choices, weakened assertions, merge, auto-merge or training
launch. Prior detailed note: git blob `6875497e56d3ca5bc041a4f4ef761ec440114d03`; raw evidence tar blob
`36d7913583675575fdeaf15969a3cb2b00d78dc5`, local
`.cache/mtg-244-ci-repair-evidence.tar.gz`.

## Engine Budget history (same runner, reference unchanged)

| Head | Search s (ref→cand) | Worst reply s | Steps/s | Result |
| --- | ---: | ---: | ---: | --- |
| `6165e0fe4` first publish | 168.192→453.756 | 4.909→15.349 | 12,817→5,655 | fail |
| `beaacaa19` | 166.656→230.730 | 4.883→7.045 | 12,939→9,904 | fail |
| `a7b0ddaff` source plans | — | — | step time +10.54% | fail |
| `68ef4a63d` lazy SBA | 164.461→132.934 | 4.818→3.860 | step time −8.60% | pass |
| `e4e527621` CI 41958999646 | 166.560→132.200 | 4.897→3.851 | 12,785→14,369 | pass |
| `c46e4f3d8` CI 42035238111 | 171.352→132.322 | 4.998→3.845 | 12,619→14,205 | pass |

All checked counts and digests match. Local timing on the shared host stays
exploratory. Each new head must earn its own CI measurement.

## Performance repairs retained

Layer nodes are discovered initially and after grants, proven-independent
instructions apply as timestamp batches, tap state has its own revision with
discovered tap reads, derived tables use slots with Arc values, operation
expressions are boxed, whole-board readers share one immutable read, and SBA
checks are lazy. Debug recomputes every cache. `prepare` fetches exact historical
commits. Source-plan detail lives in [the risk note](mtg-244-risk-validation.md).

## Head e4e527621 evidence

Layer Risk 41958999383 succeeded (artifact `.cache/mtg-244-current-risk`): all 7
policy cases, 16 payments and workloads matched. CPU float32 B16/D64, one thread,
eight samples: v8 program inference 191.312→180.609 ms, forward/backward
477.955→422.822 ms versus `fb3555ffb`; ordinary v7/v8 inference
164.485/161.084 ms, forward/backward 349.779/330.524 ms; bytes at 512 facts
v7 165360, v8 208368. Measurements only, not learning efficacy or GPU proof.
Long integration, Academy, clean-machine Play, Python Unit and conformance passed.
Failed logs: `.cache/mtg-244-current-failed.log`. The steer's
`scratch/mtg-244-current-ci.md` is in neither this checkout nor git history; the
steer text and these logs are the sources used.

## Correctness repairs after e4e527621

- Energybending, real regression: per-type mana abilities replaced the single
  modal ability. One `AddSubtypes` record now yields one ability choosing among
  its gained basic colors, in WUBRG order, preserving base choices.
- Honest Work (two tests), real regression: the source's lose-all ran after its
  grant at one timestamp. The definition now lists the set before the grant,
  matching printed order; AGENTS.md records the convention.
- Toph and Phoenix Fleet Airship, stale expectations: both asserted that removing
  an ability in layer 6 undoes its layer-4 type effect, contrary to CR 613.6 and
  the accepted Dress Down/Magus witness. Tests now assert the rule; Toph adds a
  true no-Toph baseline plus the abilityless case. The land column retains the
  granted-versus-plain proof.
- LLM text guards: colors and keyword counter timestamps already render. Native
  `program` is set only alongside `description`, which text renders. Guards list
  all four, with a directed proof of each.
- Definition audit: Honest Work is the only card pairing an attached set with an
  attached grant, and no static in a set file names an ability-loss modification, so the ordering
  repair reaches no other definition.

Check: 567 debug card tests, 169 rules tests, 65 flow lib tests, clippy and
rustfmt passed after the final `layers.rs` reduction; 60 text tests pass at
reconciliation. No local result replaces hosted evidence.

## Head c46e4f3d8 evidence

CI `42035238111`: Engine Budget, Python Unit, both conformance jobs, Clean-machine
Play, Academy Integration and long Integration pass. Academy UI fails four cases and
Integration main fails two `test_trigger_order` cases (1068 passed), both on stale
digests. Layer Risk `42035237232` fails at `Payment pool/0 differs: before`.
Budget search is 22.8% less time, worst reply 23.1% less, steps 12.6% higher.

## Digest repair evidence

- Semantic Kernel Conformance passed because its `fuzz` path validates only the
  manifest and Phase matrix; `check_root` reads `evidence.json`, in Rust Tests.
- The authored receipt predated this branch (content `28a0744e…`, 356 definitions, so
  ids shift for the 358); all 135 decisions otherwise move only in digests.
- Layer Risk: 16/16 payment receipts equal except `digest` fields, because `before`
  (`fb3555ffb`) lacked the reorder, the only card-source difference. The `before` leg
  now uses the candidate's `tla.rs`; the exact comparison stays unweakened.
  Local `--payment-only` on `201bc7b58`: all 16 match, digests included; not hosted evidence.
- Outside CI, `tests/belief/`: two `test_sampling_data` cases fail identically on
  main (message casing; left alone). `test_real_population_learning…` fails here at
  0.0926 > 0.1 and passes on main. Seeds 197/198/199: branch .093/.185/.155, main
  .156/.255/.133 (main's `.so` from Nov 4). Inconclusive at n=3; disclose at Demo,
  threshold unchanged.
