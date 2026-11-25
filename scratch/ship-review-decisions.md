# MTG-443 Ship Review Decisions (2026-11-25)

## Exact reply from Jack Heart

```text
1b 
2a
3a
4a
```

## Reading of choices (against `scratch/ship-review.html`)

- **1b:** Keep `'Tapped'` / `'Untapped'` capitalized in `academy/ui/src/lib/board/badges.ts`; update `academy/ui/e2e/mana-payment.spec.ts` assertions to `/tapped/i`; refresh `conformance/authored_match_parity.py` (`AUTHORITY_SHA256`) and `conformance/authored-match-parity-w17-v2/release-live-headless-replay-seed-0-battlefield-history.json`; proceed to `deliver`.
- **2a:** Keep as built (drop narrow-card `Double strike` finding).
- **3a:** Keep as built (width-triggered compact labels beside P/T; Attacking/Blocking in Inspect & speech only; `Struck first` kept on board).
- **4a:** Drop Ty Lee freeze duration copy finding.
- **Follow-up Tasks filed:** None.
- **Outcome:** Go on to `deliver` after applying the two CI fixes.
