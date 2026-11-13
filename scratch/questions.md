# MTG-348 assumptions (headless, 2026-11-13)

Reversible choices taken without asking; details in `etude-sdk-031.md`.

1. The branch pins etude-sdk 0.3.2 and draws one head-to-head line by
   seating (Jack Heart: "Fix etude first"; third comment: "Move every pin
   straight to 0.3.2").
2. `~/.etude/launcher.db` is never moved (Jack Heart, second Task comment);
   0.3.2 keeps `launcher-17.db` beside it.
3. `with_launcher_ends` takes the lost run `5b7d8fa7b38c` from a frozen record
   in the experiment file instead of the launcher.
4. `mtg183-elemental-eval` loses its "400 updates earlier" and "half its
   updates" curves; `two_seat` loses its cross-job direct-games chart;
   `elemental` swaps `HeadToHead` for `Crosstable`.
5. "Type checker" in the brief is read as not applicable: the repo configures
   none.
6. Kickoff, second pass: `head_to_head` takes seating names, and the frontier
   lists its 21 recorded Large-vs-Small seatings as a frozen record, because a
   tag selection is never drawn seated and `report()` sees no state. The
   alternative, teaching Etude to seat a tag, is noted and not taken.
7. Docs state 0.3.2 and `launcher-17.db` as
   shipped; the frontier's frozen names are `EQUAL_HOURS_SEATINGS` and
   `EQUAL_UPDATE_SEATINGS`.
