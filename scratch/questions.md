# MTG-348 assumptions (headless kickoff, 2026-11-13)

Reversible choices taken without asking; details in `etude-sdk-031.md`.

1. Superseded 2026-11-13: Jack Heart chose "Fix etude first"; the branch
   waits for etude-sdk 0.3.2 and draws one head-to-head line by seating.
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
