# MTG-302 plan

Built (switch `MANABOT_BUCKETED_SHAPES=1`, read in `ActorCritic.__init__`):

- `manabot/model/buckets.py`: `bucket_sizes` (one host read), `trim_observation`
  (slice + object-reference remap), `object_columns`. Need per axis = last nonzero
  row, plus every referenced row (effect CSR spans, frame ids, action
  continuation joins, object references), so only all-zero unreferenced rows go.
- Object axes (cards, permanents) trim only when not `historical_mean` and no
  belief rows. Logits padded back to full action capacity.
- `PolicyInference.prepare` measures buckets on CPU (`PreparedObservation.buckets`);
  forward trims device tensors with `trim_prepared`; the prepared dict stays full.
- Encoders take explicit admitted counts (`FactEncoder(count=)`,
  `SharedEffectEncoder(counts=)`, `ContinuationEncoder.encode_rows(count=)`), so
  malformed smaller inputs are still rejected exactly.
- Bucketed only: fact catalog ids via `record_names_masked`/`record_tokens_masked`
  (no masked selection); no CPU fact compaction; name joins via `torch.where`;
  learning encodes the full catalog.
- Kept packed/data-dependent: shared-effect host CSR routing, fact program GRU,
  legal-action `nonzero`, CPU object compaction in `_attend_objects`.

Catalog GRU decision (learner captures, 16 minibatches, B 42-99): catalog is 360
definitions / 43,704 tokens / longest 713. Referenced: 65-154 definitions,
7,540-18,710 tokens, longest 224-713 (median 383). Full catalog costs 2.3-5.8x
the tokens but ≤1.9x the serial GRU depth at the median, has one fixed packed
layout, drops the per-minibatch `unique`/`nonzero`/lengths reads, and is the
table MTG-300 caches. Chosen: full catalog.

Results (CPU, one thread, update-zero captured weights + identical 0.02 noise;
`scratch/measure_buckets.py`, `scratch/diagnose_buckets.py`, JSON in
`scratch/measure_{w64,w160l8}.json`; 300 collection batches of 256 + 8 learner
minibatches per size; learner loss is a policy-gradient + value-MSE surrogate):

| | w64 | w160l8 |
| -- | -- | -- |
| collection max abs diff logits / values | 1.3e-7 / 3.3e-7 | 3.0e-7 / 6.0e-7 |
| learner max abs diff loss / any grad element | 1.5e-8 / 2.7e-6 | 8.9e-8 / 5.3e-5 |
| grad None→zero / zero→None | none / none | none / none |
| input shape signatures, collection (of 300) | 1 → 154 | 1 → 147 |
| full signatures incl. dynamic op shapes | 300 → 300 | 300 → 300 |
| host reads/forward median, collection (prepared) | 115 → 102 | 115 → 102 |
| host reads/forward median, learner | 140 → 117 | 141 → 118 |

Largest relative gradient gaps are `policy_head.2.bias` (true gradient 0 by
softmax shift invariance; 1e-9 residues). Every captured learner batch
references definitions, so the documented None→zero case did not occur.

Padded elements, bucketed / capacity (= current learning and option 3, which
ran every lookup at capacity), w64 (w160l8 within 0.02): facts 0.198, effects
0.169, details 0.0031, members 0.031, relations 0.453, actions 0.276, events
0.389, cards 0.493, known 0.0625, decklists 0.5, permanents 0.263. Objects:
450 → median 2+128+64 = 194 slots. Current CPU inference already compacts
facts (median 62 live columns) and objects (median 136.5); bucketed medians
are 64 facts and 194 objects there. Median buckets: facts 64, effects 32,
details 16, members 64, relations 32, actions 16, events 8, cards 64, known 8,
decklists 64, permanents 32.

Checks: `pytest tests/model/test_bucketed_shapes.py` 9 passed;
11 existing model/boundary test files 162 passed, 2 skipped.

Remaining: gate; MTG-301 L4 A/B (`MANABOT_BUCKETED_SHAPES=1`).

Compress: `_ownership` reads the forward's `BucketSizes` instead of rebuilding a
partial one from shapes; fact count check uses an explicit `None` default.
`pytest tests/model/test_bucketed_shapes.py` 9 passed.
