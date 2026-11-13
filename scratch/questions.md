# Assumptions and scope

- The numerical rejection at attempted update 141 is decisive under the frozen
  rule. Skip the paid strength evaluation because it cannot admit BF16; report
  M = 75 and the absence of strength and trajectory comparisons explicitly.
- Keep `BF16_MIN_HIDDEN_DIM = 256` as the measured speed crossover, with
  `ladder_precision` returning float32 everywhere after rejection, as the Task
  specifies. Reopening BF16 admission needs another learning result.
- Use main's `model_fields_set` authoring mechanism for precision as well as
  layouts; remove the private flag and second resolver.
- Earlier A/B choices and launch history remain in git `e181224e7` and the
  experiment report. No new rental, experiment, or numerical-guard change.
