# MTG-441 ship-review notes

## Person's words (verbatim)

2026-11-25 (reply to question 5):

> 5. Unstarted MTG-442 ("Commit to unified joint combat UX in Academy live play"): 5a Close/abandon once PR #613 lands · 5b Keep open.
>
> if its fixed here, then abandon it. it is required for here though

2026-11-25 (during `/demo`, after playing Elemental GR and GW Allies):

> can you start a game where i am GW allies instead

> ok lg2m ship it

## Reading

- Verified every bullet of `MTG-442` is implemented in PR #613 (`MTG-441`): zero engine/server changes, live-play joint combat draft with raw per-creature offers folded under collapsed `Each creature's choice` (no number keys), Replay/Study unchanged, and covered by unit + Playwright E2E tests (with the 2 scripted-player E2E test-helper CI failures still to fix before landing).
- Abandoned `MTG-442` (unstarted) as superseded by `MTG-441`; joint combat remains a required part of `MTG-441`.
- After trying the built app in `/demo` (Elemental GR then GW Allies vs Search 16), Jack Heart approved shipping as-is with no extra build pass or follow-up Tasks filed (questions 2–4 stay at their built defaults / dropped). Proceeding from `ship-review` to `deliver` (`reflect` → `gate` → `land --wait-and-fix`).
