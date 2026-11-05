# MTG-244 retained risk evidence

Full note: `1930146b08a1f1481af6e3b46a20f87d4a26a8fa:scratch/mtg-244-risk-validation.md`.
See [integrated context](mtg-244-layers.md) and this Task's active
[risk acceptance](mtg-241-risk-validation.md). This archival pointer replaces the
sibling's ambient plan, not its outstanding work.

Historical component mitigations: whole-buffer GRU replaced by independent packed
programs (ordinary CPU inference 15.36 to 3.94 ms, two-program 15.61 to 4.76 ms);
signed log magnitudes distinguish 4096/8192; batched shared dispatch reduces GRU
calls; mana-independent pool writes reduce 471 allocations to 10, while required
tap invalidation remains. Complete real-payment allocation counts fell 1176 to 510
(floating RR) and 2882 to 1668 (land-funded). These are bounded component results,
not combined-head speed or learning-quality evidence. Exact bundles, harnesses,
revisions and limits are preserved in the full note.

V8 default arrays are 208368 bytes, versus v7/128 facts 146928 bytes. No capacity
was reduced. Training efficacy, accelerators and generalization are unmeasured.
The first published Engine Budget regression and failed reference fetch supersede
any earlier inference of readiness. MTG-244 owns that repair; MTG-241 preserves
it as contrary integration evidence and retains combined-head measurement work.
