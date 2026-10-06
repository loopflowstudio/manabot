# Large legal-choice navigation

ETU-77 software candidate, 2026-10-05. Human usability acceptance remains open.

ActionPanel offers **Find an action** when the current list has at least eight
choices. It matches card/action label text, ignoring case and surrounding spaces.
All offers remain visible until a filter is entered. The count describes the
current panel list; a board selection may already have narrowed that list.
**Clear filter**, or Escape in the search field, restores the list. Navigation
sends no Command. Selecting a result sends its original offer through the
existing revision-bound Command path. Recovery, a new update, board selection,
and Learn-mode navigation clear the filter. Duplicate labels remain separate
choices; focusing or hovering a result highlights its original board objects.

Learn keeps its accepted Take a Lesson / Discard and draw / Decline interaction,
including previews and Back. Costs and other small choices remain direct.
The eight-choice threshold and text filtering are reversible product hypotheses;
no preference or approval by Jack Heart is implied.

## Retained positions and coverage

[The fixture](../etude/fixtures/choice-navigation.json) pins source commit,
configuration, every prefix offer ID/label, native digest and full viewer frame.
These are actual ordinary GameSession positions in UR Lessons versus GW Allies,
with Random opponents and auto-pass off. They are newly captured software
fixtures, not the unidentified original awkward human position or scored games.

| Family | Hero / seed | Prefix Commands / revision | Choices | Intended review action |
|---|---|---|---|---|
| Target | UR / 3 | 62 / 109 | 9 | Target Earth Kingdom Jailer; also locate each of three distinct Allies |
| Cleanup discard | GW / 62 | 11 / 30 | 8 | Discard Kyoshi Warriors; locate both Earth Kingdom Jailer copies |
| Learn | UR / 51 | 24 / 54 | 10 | Take It'll Quench Ya!; cancel discard navigation before selecting a Lesson |
| Block | UR / 3 | 91 / 176 | 6 | Block Suki with Tiger-Seal, or choose not to block |
| Optional cost | UR / 0 | 43 / 87 | 2 | Pay Firebending Lesson's cost, or decline |

Native tests reconstruct all five prefixes and compare exact state and offers;
stale Commands leave the state unchanged. Browser tests retain every matching
ID, including duplicate labels, exercise keyboard activation and recovery, and
execute target/discard choices against the real backend. Existing Learn browser
checks exercise all three outcomes, previews, Back and reconnect. Existing combat
checks cover attacks and blocks by keyboard and pointer. The release matrix
covers priority, look/select, scry and waterbend in both deck assignments; no new
rare waterbend fixture was captured here. MODAL and LEGEND_RULE remain outside
that selected-matchup matrix. This change neither extends rules support nor
changes ETU-107's model-facing contract.

## Local review walkthrough

After `./scripts/play` has installed the locked local runtime, stop that instance.
Run from the repository root (choose unused ports):

```bash
ETUDE_REVIEW_CHOICE=CHOOSE_TARGET ETUDE_API_PORT=8147 ETUDE_FRONTEND_PORT=5347 \
  ETUDE_TRACES_DIR=/tmp/etude-choice-review ETUDE_PLAY_RECORD_ORIGIN=automated_validation \
  npm --prefix frontend run test:e2e -- choice-navigation.spec.ts --headed -g 'live retained'
```

The test replays the retained prefix through the real server and pauses with the
nine-target decision ready. Play in that browser while the inspector is paused;
resume the inspector to finish. Repeat with `ETUDE_REVIEW_CHOICE=DISCARD` for the
eight-card cleanup. These sandbox attempts have automated provenance because
the prefix is scripted; they do not count as a human completion cohort.
The ordinary `./scripts/play --demo learn` path remains the Learn walkthrough.

Review intended choices before discussing controls: can Jack Heart find and
execute the named play without coaching? Try a nonexistent name, cancel the
filter, use keyboard-only selection, and revisit after another game. Record
stopped attempts as well as successes, along with the exact fixture and chosen
offer. Search versus grouping, the threshold, and duplicate-copy clarity remain
open product decisions. No broad full-game usability claim follows from these
checks. ETU-77 stays open; ETU-75 retains its empirical Learn acceptance limits.
