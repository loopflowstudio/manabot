# Frozen smoke observation tensors

`replay.json` retains all 303 Commands from the sealed w13 mirror smoke attempt
`f38ccc9074d3935643f0e255dec6bf51eb94e713bcffc3c297c964ba6ffd0b9b`, its exact
match/observation configuration, actor order and provenance hashes.
`encoded.npz` holds every pre-Command actor-view encoding and the final
observation returned by the last Command (304 rows per tensor key).

Captured with the unchanged Python encoder at
`7fd6c202f163bc7dd0d507c7eab121086af84a68:manabot/env/observation.py`, before the
direct-row-write optimization. Capture verified every pending decision frame
and semantic actor observation against the sealed attempt. Search receipts and
the trained checkpoint are unnecessary for this regression replay. It includes
both seats, hidden hands, payment decisions, populated permanents, padding,
action focus and the terminal boundary. The test checks shape, dtype, every
element and raw bytes, rather than numerical tolerance.

Do not regenerate expected tensors from the optimized implementation. For a
deliberate observation-contract change, retain this provenance when replacing
the fixture; it is evidence for the frozen world, not a new policy input schema.
