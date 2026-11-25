# MTG-439 ship-review notes

## Jack Heart — 2026-11-25 09:39

> 1. Pass 2 also skips identical basics-only payments (Mountain ×4 for {2}{R}, Plains ×3 for {W}{W}): drops benchmark steps by 3,698 (118,559 → 114,861, −3.1%), causes
>   most of the 7 red CI jobs on PR #616 (3 Rust card tests, 25 Python integration tests, authored match parity revisions 172→168, 1 Academy UI E2E test), and Engine Budget
>   search total jumped 112.505 s → 127.642 s (+13.5%, over the +10% cap — needs investigation before adopting baseline.json).
>
> We should investigate this. this should make search go FASTER if done right

## Jack Heart — 2026-11-25 09:55

> yes
> can you explain 2-5

## Jack Heart — 2026-11-25 09:58

> 2a 3b 4b change 2 to Tap Kyoshi Village for {G}/{W 5c
