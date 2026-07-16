# Search branching decision v1

Decision: **retain compact full clone as the production default**.

Do not integrate either optimized representation. Keep both as conformance/benchmark drivers and remove them from production hot paths.

The decision applies the pre-registered whole-rollout thresholds in `dense-page-cow-prereg-v1.md`. Correctness and matched provenance are absolute gates; clone latency is diagnostic only.

## Matched primary evidence

| Driver | Cell | sims/s | vs full | p99 root | vs full | peak RSS | vs full | RSS delta | vs full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full clone | `flat-saturated-64-v1` | 2001.7 | 1.000x | 0.379s | 1.000x | 128.7 MiB | 1.000x | 2.7 MiB | 1.000x |
| full clone | `flat-single-64-v1` | 294.5 | 1.000x | 0.258s | 1.000x | 15.9 MiB | 1.000x | 0.9 MiB | 1.000x |
| full clone | `retained-saturated-16-v1` | 350.6 | 1.000x | 0.125s | 1.000x | 81.7 MiB | 1.000x | 29.9 MiB | 1.000x |
| full clone | `retained-single-8-v1` | 321.7 | 1.000x | 0.010s | 1.000x | 14.9 MiB | 1.000x | 3.5 MiB | 1.000x |
| clone + undo | `flat-saturated-64-v1` | 1823.4 | 0.911x | 0.487s | 1.286x | 193.2 MiB | 1.502x | 22.1 MiB | 8.329x |
| clone + undo | `flat-single-64-v1` | 278.7 | 0.946x | 0.335s | 1.302x | 24.6 MiB | 1.546x | 4.9 MiB | 5.509x |
| clone + undo | `retained-saturated-16-v1` | 350.3 | 0.999x | 0.135s | 1.078x | 84.2 MiB | 1.030x | 30.7 MiB | 1.025x |
| clone + undo | `retained-single-8-v1` | 318.3 | 0.989x | 0.010s | 1.080x | 16.6 MiB | 1.111x | 4.7 MiB | 1.354x |
| event-page COW + undo | `flat-saturated-64-v1` | 1686.8 | 0.843x | 0.535s | 1.413x | 193.2 MiB | 1.501x | 22.0 MiB | 8.288x |
| event-page COW + undo | `flat-single-64-v1` | 272.7 | 0.926x | 0.357s | 1.387x | 24.2 MiB | 1.521x | 5.0 MiB | 5.579x |
| event-page COW + undo | `retained-saturated-16-v1` | 331.7 | 0.946x | 0.203s | 1.621x | 37.7 MiB | 0.462x | 2.1 MiB | 0.070x |
| event-page COW + undo | `retained-single-8-v1` | 311.5 | 0.968x | 0.011s | 1.175x | 13.9 MiB | 0.929x | 3.2 MiB | 0.915x |

## Threshold results

- Clone-plus-undo sequential bar: `fail`.
- Page-COW retained-memory bar: `fail`.
- Page-COW general-driver bar: `fail`.

Clone plus undo is rejected unless both flat cells reach 1.20x full-clone throughput without exceeding 1.10x p99 latency or RSS. Page COW is rejected for retained use unless both retained cells preserve at least 0.90x throughput and meet every registered RSS reduction; general use also requires the flat guardrails. The table above makes each failed comparison explicit.

## Branch lifecycle diagnostics

| Driver | eager full forks | full checkpoints | max journal peak | max COW peak |
|---|---:|---:|---:|---:|
| full clone | 4416 | 27648 | null | null |
| clone + undo | 4416 | 0 | 30.23 MiB | null |
| event-page COW + undo | 0 | 0 | 30.23 MiB | 1.01 MiB |

Full-clone and clone-plus-undo eager counts represent deep outer forks; only full clone takes full mark snapshots. Page COW reports zero full logical-state forks and checkpoints, with numeric copied-page peaks. Unsupported allocator totals remain null in the raw receipts.

## Provenance

- `full_clone/current_game_v1`: `experiments/data/w2-182-search-branching-v1.json`; artifact `0d623a67b186e9c138cbee99ab14cc0df948d30e03ec999c52b7394818af90f0`.
- `compact_clone_undo/current_game_v1`: `experiments/data/w2-198-compact-clone-undo-v1.json`; artifact `389bd14afcb925489b5a29f7c0dacb34baa388886e8f3eb0cad13aa9fa1bb395`.
- `dense_page_cow_undo/event_pages_4k_v1`: `experiments/data/w2-199-dense-page-cow-undo-v1.json`; artifact `707d346625a198af57b0b24af9237969b7fc5ea9bb84979ed8e80cc3db33dcd3`.

All three receipts use source `4354b41a80706409cdc23b26979ca2679f88b62a8d48516e18fc7b9cd3ab8beb`, revision `2d804a766757cace5c3e0d41c3a95f36b7e31cc6`, and release binary `af197d6d7f8297cdf2cdf72a2b417af328310cac33ef5f78bffaa6e2a5e80a83` on the identical recorded host.
Logical fixture summaries, seeds, workload dimensions, result checksums, outcomes, caps, and sampled final hashes match exactly across candidates.
