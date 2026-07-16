# Dense event-page COW branching preregistration v1

This preregistration fixes the W2-199 selection rules before the canonical
three-driver timing run. Correctness under `manabot.search-branching.v1` and
matched source, revision, binary, host, fixtures, seeds, workloads, logical
checksums, outcomes, caps, and final hashes are absolute gates. Clone latency
is diagnostic and cannot select a representation.

## Candidate-specific bars

Clone plus undo is eligible for sequential flat rollouts only when both flat
cells are at least 20% faster than full clone in simulations per second, while
neither p99 root latency nor absolute peak RSS is more than 10% worse.

Dense 4 KiB event-page COW plus undo is a meaningful retained-memory win only
when both retained cells stay within 10% of full-clone throughput and neither
has worse absolute peak RSS. In addition:

- `retained-saturated-16-v1` must reduce absolute peak RSS by at least 15% and
  peak RSS delta by at least 40%; and
- `retained-single-8-v1` must reduce peak RSS delta by at least 25%.

Page COW is eligible as the general driver only after clearing the retained
bar and keeping every flat cell within 10% of full clone for simulations per
second, p99 root latency, and absolute peak RSS.

## Selection

If page COW clears both retained and flat bars, select it for all measured
branching shapes. If it clears only the retained bar, select a workload-specific
hybrid: page COW for simultaneously retained slots and the eligible sequential
driver (clone plus undo when it clears its bar, otherwise full clone) for flat
rollouts. If only clone plus undo clears its bar, use it for sequential flat
rollouts and retain full clone for retained slots. If no optimized candidate
clears its workload-specific bar, retain compact full clone as the production
default.

Failed candidates remain conformance and benchmark implementations. Thresholds
will not be tuned and page coverage will not be broadened after seeing the
canonical result.
