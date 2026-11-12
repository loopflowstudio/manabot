# Assumptions (reversible)

- Old-format runs (written 1659b0e29..this change, one day on main) use a documented cutoff rather than an Etude conversion: reading them raises, naming `bf4e280f7`. Simpler than adding conversion to Etude; switch if retained runs of that format turn out to matter.
- New table name `self_play_outcomes` (distinct from the old `self_play_outcomes/table` metric so old runs are detected, not silently empty).
- Out of scope, seen while gating: `tests/training/test_training_throughput.py::test_a_packed_seed_digests_as_it_does_solo` fails on `self-play*.pt/source/etude_run`. Since #464 (`6cac46c1a`) the checkpoint source records `ETUDE_RUN`; the test names its solo and packed workers differently, and `experiments/training_throughput_packing.py::_VOLATILE_POLICY` only excludes `bc/run_id`. Not caused by the outcome-table change; candidate bug Task.
