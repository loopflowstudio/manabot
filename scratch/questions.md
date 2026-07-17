# Confirmed interpretation and assumptions

- The canonical Game index is one complete, globally chronological index over
  every deliberate player decision in the pinned match. Human commands and
  opponent-policy choices are indexed. Configured auto-passes, F6 expansion,
  and rules resolution are not decisions and remain semantic continuation.
- Every decision row is scoped to its acting viewer and stores only the exact
  `ExperienceFrame`, `InteractionOffer`, and `Command` safe for that viewer.
  The complete mixed-view index is authority-private. Any client or Study
  artifact is an authorized single-viewer projection and can never combine
  both players' private frames or presentation tracks.
- A stable address is stable within an immutable replay ID and exact decision
  payload. Editing the stored replay or row invalidates its integrity binding
  instead of silently resolving the address against changed history.
- Legacy traces that did not persist authoritative frames, offers, commands,
  viewer identities, revisions, and presentation cursors are not
  canonicalizable. They remain usable by the legacy replay viewer, but the
  canonical resolver fails closed instead of inferring missing history.

- The first fork-and-return slice retains exact engine roots only for the live
  completed `GameSession`. Durable root persistence or reconstruction is a
  later Game storage decision; this provider never rebuilds authority from
  legacy trace actions.
