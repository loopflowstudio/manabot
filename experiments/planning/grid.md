# Strongest playable manabot: planning grid

Planning envelopes only. Volumes and strength remain unknown without matching ETU-79 receipts.
Paid quotes are dated references; reserve is protected. No execution is authorized.

| Window | Cash cap | Choice | Scheduled hours | Host-hours | Cash range | Reserve | Unspent above reserve |
|---|---:|---|---:|---:|---:|---:|---:|
| 10min | $9.99 | local | 0.167 | 0.167 | $0.005–0.015 | $1.998 | $7.977 |
| 10min | $1000.00 | local | 0.167 | 0.167 | $0.005–0.015 | $200.000 | $799.985 |
| 1hr | $9.99 | local | 1.000 | 1.000 | $0.030–0.090 | $1.998 | $7.902 |
| 1hr | $1000.00 | local | 1.000 | 1.000 | $0.030–0.090 | $200.000 | $799.910 |
| 1day | $9.99 | local | 20.333 | 20.333 | $0.610–1.830 | $1.998 | $6.162 |
| 1day | $1000.00 | local | 20.333 | 20.333 | $0.610–1.830 | $200.000 | $798.170 |
| 1week | $9.99 | local | 48.000 | 48.000 | $1.440–4.320 | $1.998 | $3.672 |
| 1week | $1000.00 | local | 48.000 | 48.000 | $1.440–4.320 | $200.000 | $795.680 |

## 10min / near_zero

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 180 s. Cash: [0.005, 0.015]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $1.998; unspent above reserve $7.977.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | None |
| generation | M4-Max-128GiB × 1 × 4 | 180.00 | 0.0500 | 0.0500 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 60.00 | 0.0167 | 0.0167 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 120.00 | 0.0333 | 0.0333 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 10min / paid

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 180 s. Cash: [0.005, 0.015]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $200.000; unspent above reserve $799.985.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | None |
| generation | M4-Max-128GiB × 1 × 4 | 180.00 | 0.0500 | 0.0500 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 60.00 | 0.0167 | 0.0167 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 120.00 | 0.0333 | 0.0333 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

### local_plus_cloud: INFEASIBLE

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 512.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 3240 s. Cash: [0.0, 0.0]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $200.000; unspent above reserve $800.000.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- startup/export, local availability or fixed cash consumes the envelope

## 1hr / near_zero

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 420 s. Cash: [0.03, 0.09]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $1.998; unspent above reserve $7.902.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| generation | M4-Max-128GiB × 1 × 4 | 1500.00 | 0.4167 | 0.4167 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 780.00 | 0.2167 | 0.2167 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 1hr / paid

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 420 s. Cash: [0.03, 0.09]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $200.000; unspent above reserve $799.910.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| generation | M4-Max-128GiB × 1 × 4 | 1500.00 | 0.4167 | 0.4167 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 780.00 | 0.2167 | 0.2167 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

### local_plus_cloud: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 512.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 3480 s. Cash: [3.1483999999999996, 3.2664999999999997]; storage $1.000, transfer $1.000, tax $0.286–0.296 included.
Reserve $200.000; unspent above reserve $796.734.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| generation_startup_transfer | CCX33 × 1 × 4 | 1020.00 | 0.2833 | 0.2833 | None |
| generation | CCX33 × 1 × 4 | 56.60 | 0.0157 | 0.7167 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training_startup_transfer | Lambda-A6000 × 1 × 1 | 1020.00 | 0.2833 | 0.2833 | None |
| training | Lambda-A6000 × 1 × 1 | 33.96 | 0.0094 | 0.0167 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 120.00 | 0.0333 | 0.0333 | None |
| evaluation_startup_transfer | CCX33 × 1 × 4 | 1020.00 | 0.2833 | 0.2833 | None |
| evaluation | CCX33 × 1 × 4 | 29.43 | 0.0082 | 0.7167 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 1day / near_zero

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 1200 s. Cash: [0.61, 1.8299999999999998]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $1.998; unspent above reserve $6.162.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation | M4-Max-128GiB × 1 × 4 | 39600.00 | 11.0000 | 11.0000 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 18000.00 | 5.0000 | 5.0000 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 14400.00 | 4.0000 | 4.0000 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 1day / paid

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 1200 s. Cash: [0.61, 1.8299999999999998]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $200.000; unspent above reserve $798.170.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation | M4-Max-128GiB × 1 × 4 | 39600.00 | 11.0000 | 11.0000 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 18000.00 | 5.0000 | 5.0000 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 14400.00 | 4.0000 | 4.0000 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

### local_plus_cloud: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 512.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Repeat independent training seeds only if measured complete loops fit
Fixed overhead: 4260 s. Cash: [18.493116666666666, 20.424]; storage $1.000, transfer $1.000, tax $1.680–1.854 included.
Reserve $200.000; unspent above reserve $779.576.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation_startup_transfer | CCX33 × 2 × 4 | 1020.00 | 0.5667 | 0.5667 | None |
| generation | CCX33 × 2 × 4 | 39600.00 | 22.0000 | 23.4333 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training_startup_transfer | Lambda-A6000 × 1 × 1 | 1020.00 | 0.2833 | 0.2833 | None |
| training | Lambda-A6000 × 1 × 1 | 18000.00 | 5.0000 | 5.0000 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation_startup_transfer | CCX33 × 2 × 4 | 1020.00 | 0.5667 | 0.5667 | None |
| evaluation | CCX33 × 2 × 4 | 14400.00 | 8.0000 | 9.4333 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 1week / near_zero

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Compare scratch self-play and distillation+self-play at matched total cost after profiling; warm-start cost included
Fixed overhead: 1200 s. Cash: [1.44, 4.32]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $1.998; unspent above reserve $3.672.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation | M4-Max-128GiB × 1 × 4 | 73542.86 | 20.4286 | 20.4286 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 61285.71 | 17.0238 | 17.0238 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 36771.43 | 10.2143 | 10.2143 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

## 1week / paid

Local first; no measured benefit justifies rental spending

### local: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 128.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Compare scratch self-play and distillation+self-play at matched total cost after profiling; warm-start cost included
Fixed overhead: 1200 s. Cash: [1.44, 4.32]; storage $0.000, transfer $0.000, tax $0.000–0.000 included.
Reserve $200.000; unspent above reserve $795.680.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation | M4-Max-128GiB × 1 × 4 | 73542.86 | 20.4286 | 20.4286 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training | M4-Max-128GiB × 1 × 1 | 61285.71 | 17.0238 | 17.0238 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation | M4-Max-128GiB × 1 × 4 | 36771.43 | 10.2143 | 10.2143 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

### local_plus_cloud: MEASUREMENT_REQUIRED

Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished
Recipe: search_distillation, width 64 / 4 attention heads, batch 512.
Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.
Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling.
Compare scratch self-play and distillation+self-play at matched total cost after profiling; warm-start cost included
Fixed overhead: 4260 s. Cash: [48.767316666666666, 54.083999999999996]; storage $1.000, transfer $1.000, tax $4.432–4.914 included.
Reserve $200.000; unspent above reserve $745.916.

| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |
|---|---|---:|---:|---:|---|
| setup | M4-Max-128GiB × 1 × 1 | 900.00 | 0.2500 | 0.2500 | None |
| generation_startup_transfer | CCX33 × 2 × 4 | 1020.00 | 0.5667 | 0.5667 | None |
| generation | CCX33 × 2 × 4 | 86400.00 | 48.0000 | 49.4333 | {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None} |
| training_startup_transfer | Lambda-A6000 × 1 × 1 | 1020.00 | 0.2833 | 0.2833 | None |
| training | Lambda-A6000 × 1 × 1 | 72000.00 | 20.0000 | 20.0000 | {'training_exposures': None} |
| export | M4-Max-128GiB × 1 × 1 | 300.00 | 0.0833 | 0.0833 | None |
| evaluation_startup_transfer | CCX33 × 2 × 4 | 1020.00 | 0.5667 | 0.5667 | None |
| evaluation | CCX33 × 2 × 4 | 43200.00 | 24.0000 | 25.4333 | {'scored_paired_deals': None} |

Volumes: {'accepted_labels': None, 'accepted_training_rows': None, 'complete_games': None, 'training_exposures': None, 'scored_paired_deals': None}. Strength: unknown.
Largest uncertainty: complete-loop feasibility and quality on the corrected world.
Next measurement: ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles.

- generation: ETU-79 world/pipeline/config binding pending
- training: ETU-79 world/pipeline/config binding pending
- evaluation: ETU-79 world/pipeline/config binding pending

Inputs and dated reference:

```json
{
  "inputs": {
    "local_hardware": "M4-Max-128GiB",
    "local_hours": 48,
    "electricity_low": 0.03,
    "electricity_high": 0.09,
    "local_workers": 4,
    "setup_seconds": null,
    "cloud_startup_seconds": 900,
    "transfer_seconds": 120,
    "storage_usd": 1,
    "transfer_usd": 1,
    "tax_rate": 0.1,
    "inference_mode": "policy_only",
    "inference_device": "cpu",
    "inference_p95_ms": 100,
    "inference_rss_gib": 1,
    "inference_search_simulations": 0,
    "world_identity": "pending",
    "pipeline_identity": "pending",
    "config_identity": "pending",
    "preparation": "Existing Mac; setup allowance includes environment check. No admitted dataset assumed."
  },
  "price_reference": {
    "sha256": "253c42009598bab82eadc649cf5d22c8f56df1fa5c5e9b7efcabc08acfebfde3",
    "observed_date": "2026-09-24",
    "sources": [
      "https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/",
      "https://lambda.ai/pricing"
    ],
    "billing": {
      "cpu": {
        "increment_seconds": 3600,
        "source": "https://docs.hetzner.com/cloud/billing/faq/"
      },
      "gpu": {
        "increment_seconds": 60,
        "source": "https://docs.lambda.ai/public-cloud/billing/"
      }
    },
    "requote_before_execution": true
  },
  "evidence_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
}
```

