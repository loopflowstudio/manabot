# MTG-216: closed-frame fix, pre-merge check (2026-11-02)

Direction (Jack Heart): fix the closed-game Academy gap in this PR.

Outcome: fix forward, in PR #408, before merge. Not a post-merge verification.

Checked in headless Chromium against the real server (Random, Closed, New Game):
panel "Decklist closed"; bar "Closed decklist"; sampled opponent names absent from
page text and WebSocket frames (opening frames); card art renders.
Tests: closed frames, trace export, rematch stays closed, full closed game.

Remaining: `/api/traces` shows the villain's deck (`gw_allies`) during a live closed
game. Left unchanged and raised with Jack Heart.
Unrelated: `tests/academy/test_ur_lessons_increment.py` hard-codes permanent width 37;
main is 38. Not run by CI, not touched by this PR.
