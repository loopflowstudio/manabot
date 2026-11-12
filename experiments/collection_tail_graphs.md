# Removing the host-bound small-call tail of collection on an L4 — MTG-318

**Committed before any scored run.** *Question:* can CUDA graphs, through
`torch.compile(mode="reduce-overhead")` or a hand capture per bucket signature,
or coalescing concurrent small requests at the collector, remove the per-call
host cost of collection's small policy calls? In the
[L4 step profile](l4_step_profile.md), calls of eight rows or fewer were 1,385
of 1,717 calls in a w160l8 update. Each took 16–27 ms for about 5 ms of GPU
work, and bucketed shapes made them 1.38× slower. *Prediction:* compilation
breaks the graph at each remaining compaction read, so it gains less than 1.5×.
A hand capture per bucket signature brings bucketed calls of eight rows or fewer
from about 25 ms to 8 ms or less (≥ 3×) at both w160l8 and w384l8. Coalescing
removes under 20% of tail calls, because the tail is mostly one game left per
producer rather than many concurrent small requests. Graphs are therefore the
better lever. *Refuted if* no graph path reaches 1.5× on calls of eight rows or
fewer at both sizes, or coalescing removes more tail time than the best graph
path. *Cost cap:* USD 5 under the launcher's `--max-usd`.

**Decision rule.** A path is accepted if, on the retained real collection
inputs replayed on the L4, it meets all of these:

1. Median per-call latency for calls of eight rows or fewer is at least 1.5×
   faster than eager with bucketed shapes, at both w160l8 and w384l8.
2. Calls of 9–32 rows are no slower than eager bucketed (within 5%).
3. Logits, values and log-probabilities match eager bucketed within 1e-5
   (maximum absolute difference) on every replayed input.
4. Projected whole-collection time per w160l8 update falls, counting capture or
   compilation cost spread over a 100-update run.

An accepted path ships as a typed `Execution` field, omitted from the
serialized regime when unset, with tests that pin its outputs to eager on
retained minibatches. If no path is accepted, this report is the result and no
code changes.

## What is measured

Arms, each at w160l8 and w384l8 on the same retained inputs as the step profile,
in interleaved order with a baseline repeat at the end:

| Arm | Forward |
|---|---|
| eager | default shapes |
| eager bucketed | `Execution(bucketed_shapes=True)` |
| compiled | eager bucketed under `torch.compile(mode="reduce-overhead")` |
| captured | eager bucketed with one CUDA graph per bucket signature (when compilation leaves breaks) |
| coalesced | eager bucketed, with the collector merging concurrent small requests |

Each arm reports:

- per-call latency and GPU busy time by bucket signature, grouped as calls of
  eight rows or fewer and calls of 32 rows or fewer;
- whole-collection time per update, projected from each signature's latency and
  the in-situ call mix, and for coalescing measured as the number of calls made;
- graph breaks and recompilations under compilation, each attributed to the
  host read that caused it;
- the remaining data-dependent host reads in the bucketed forward (nonzero legal
  rows, used columns, program lengths, CSR routes and packed GRU lengths), with
  what removing each would cost;
- one-time capture or compilation cost, and memory held by captured graphs.

*Rollout recording*, the 76–172 s of update 1 that the step profile left
unexplained, is timed by sub-phase over several updates of the same in-situ job.
It is explained if its sub-phases add up to within 10% of the phase. If not,
the report bounds it.
