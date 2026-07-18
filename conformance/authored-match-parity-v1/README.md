# Authored Match Parity v1

`release-live-headless-replay-seed-0.json` is the checked RUL-5 proof for the
fixed PR #136 UR Lessons versus GW Allies command tape. It compares all 133
state checkpoints and all 132 ordered semantic consequence groups across the
release WebSocket boundary, direct managym play, and persisted canonical replay.
Both player projections are checked at every checkpoint, the two canonical
presentation tracks preserve all 61 ordered events without leaking non-actor
command identities, and an unadmitted spectator fails closed.

The same receipt captures the real revision-35 public candidate for object
`102@2`. At revision 37, a current Command carrying its authority-private exact
binding returns typed `stale_object`; the retained revision-35 Command returns
typed `stale_revision`. Both rejections preserve the state witness and semantic
event cursor.

Run `./scripts/verify-authored-match-parity` from any checkout of the recorded
source closure. The command rebuilds the CPython 3.12 extension, runs the
focused Rust proof in debug, and recomputes the receipt through uv.
