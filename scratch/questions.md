# MTG-348 assumptions (headless kickoff, 2026-11-13)

Reversible choices taken without asking; details in `etude-sdk-031.md`.

1. The frontier's two head-to-head charts will show both jobs' lines under
   0.3.1 (a mirrored Small line). Chosen over re-seating a frozen experiment.
   Contradicts the 2026-11-11 "no mirrored head-to-head line" request until
   Etude can draw a fixed pair once.
2. `~/.etude/launcher.db` (schema 13, in use) is left where it is. 0.3.1
   refuses it, so launch and status commands from this checkout need a person to
   move it aside first.
3. `with_launcher_ends` takes the lost run `5b7d8fa7b38c` from a frozen record
   in the experiment file instead of the launcher.
4. `mtg183-elemental-eval` loses its "400 updates earlier" and "half its
   updates" curves; `two_seat` loses its cross-job direct-games chart;
   `elemental` swaps `HeadToHead` for `Crosstable`.
5. "Type checker" in the brief is read as not applicable: the repo configures
   none.
