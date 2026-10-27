# Editable Experiment report demo

Open [comparison.html](comparison.html) to read the report. Edit and Run All in
[report-generator.ipynb](report-generator.ipynb) to regenerate it from retained
artifacts. The notebook contains editable data discovery and plotting code;
figures appear only in HTML. No training or tracker access occurs.

Jack Heart's edited `comparison.ipynb` and `.ipynb_checkpoints` are preserved in
place and backed up byte-for-byte in `.runs/etu113-notebook-preserved`. The new
filename intentionally avoids overwriting that original generator. The original
manifest binds the historical export, not Jack Heart's later notebook edit.
`report-manifest.json` binds the revised generator/dashboard and unchanged inputs.
The default report has matched time/work curves, RL objective and sampled RSS;
[metric documentation](../../../docs/experiment-metrics.md) explains deeper analysis.

This is a tiny software fixture, not a scientific comparison. Four single-update
TrainingRuns produced four raw milestone cohorts and 16 exact-replayed games.
The final execution receipt records 50.78 elapsed seconds and 61.30 additive
learner/evaluator process seconds, including 47.17 evaluator seconds. Notebook
execution was a separate headless acceptance check. One monitoring deal per cohort
cannot establish useful uncertainty or strength. No defaults or scientific plans
changed. Earlier process/game timeout attempts remain in the Task checkout's
`.runs/etu113-validation/{initial-timeout,game-timeout}`.

The original notebook/record exports were copied byte-for-byte from the final gate
fixture; `manifest.json` pins their bytes. Absolute source/checkpoint/trace paths
inside them retain their original provenance. Model binaries, SQLite and command
tapes remain in the original Task checkout under `.runs/etu113-validation/gate`;
they are unnecessary for notebook plotting and are not republished here. Copying
the exports does not re-admit a checkpoint or claim a fresh replay.

The editable declaration is
[experiment_demo.py](../../runners/experiment_demo.py). The shared runner generates
this same notebook for ordinary executions. See the
[execution guide](../../../docs/experiment-execution.md) for launch, placement,
continuation, budget and interpretation contracts.
