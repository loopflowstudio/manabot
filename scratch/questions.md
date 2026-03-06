# SPS closeout assumptions

- Reproduced the historical baseline only for the quick 16-env env-only case at
  commit `4c55c4e501e729c9552859cdcd29d9e57d3e4aa9`. I did not rerun long
  training/inference benchmarks on that old commit because the closeout design
  explicitly treated baseline reproduction as optional beyond quick comparisons.
