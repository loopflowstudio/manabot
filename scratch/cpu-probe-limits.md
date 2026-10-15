# Original CPU probe limits — retained protocol

The complete pre-execution protocol and refinements are preserved byte-for-byte
in `cpu-probes/b913dd906459f34a339d1afe4de1b208e9127a142951b65756dd14e9472035c4.limits.txt`.
Original setup cap600wallseconds; original probe/correctness cap240summed command
seconds,90perattempt,6GiB sampled RSS, one Torch/BLAS thread. Setup and measured
work are separate. This remains the legacy wrapper's240second cap; the later
accelerator protocol separately allows300measurement/180testseconds.

Initial no-paid and no-checkpoint restrictions applied to the first CPU report;
Jack Heart later authorized a total$50 GPU validation budget. See gpu-validation.md
for the bounded allocation history and cost ledger. Historical timeouts, failed
source admission and unmatched evidence remain retained, not reclassified.
