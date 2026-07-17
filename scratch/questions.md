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

# INT-2 assumptions carried into implementation

- The requested combat frontier is satisfied by complete atomic attacker
  declarations. Blocker and mid-resolution decisions remain a fixed,
  arm-identical executor and are counted explicitly; they are not presented as
  learned competencies.
- Lightning Bolt is added through a separately named experimental semantic
  source and IR. The production `two_deck.source.json` remains the exact UR/GW
  product artifact.
- The first supervised labels use a bounded deterministic engine oracle. Search
  is introduced only if a measured policy failure leaves label quality and
  representation quality confounded.
- The first implementation slice is the reusable real-engine decision seam.
  It proves the 35-target frontier with admitted Igneous Inspiration and 33
  admitted permanents; the separate Lightning Bolt IR and holdout audit remain
  a later artifact slice rather than being coupled to runtime binding.

# Authored-match assumptions

- The task seed is the acceptance contract because this branch has no existing
  scratch design document.
- “On compiled semantics” means the exact UR Lessons versus GW Allies deck
  pairing must construct its live `Game` from the checked-in typed IR. Other
  deck configurations retain the broader legacy content pack.
- The compiled pack may reuse the existing registry's oracle text strictly as
  presentation copy; all rules-relevant characteristics, costs, targets,
  triggers, and effects come from the typed IR.
