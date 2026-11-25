# MTG-441 questions

- **Marquee over lands/cards.** The brief says "marquee/drag-across selection
  for drafting multiple attackers (or selecting lands/cards)". The plan
  builds the marquee for the attack draft only. Tapping a land or choosing a
  card is one committed Command each, with no draft and no preview that
  predicts the next frame, so a marquee there would chain irreversible
  commits. Assumed out of scope; a multi-land marquee would need a shared
  "payment draft" at the authority first.
- **Cast and target in one drag** (drop a hand card straight onto its
  target). Left out for the same reason: two Commands with an unpredicted
  frame between them. Possible later with a native cast preview.
- **Drag to take a block back** (drag a drafted blocker off into empty
  space). Left out: indistinguishable from cancelling a drag. Click still
  takes it back.
- **Marquee only adds, starts in battlefield `.rows`, and highlights caught
  creatures live.** No modifier to subtract ("Clear the draft" and clicks
  remove); plates and hand do not start marquees; intersecting declarable
  creatures highlight while sweeping so the selection is visible before release.
- **Blocker drag derives placement instead of mutating `armed`, and hides the
  dragged blocker's old standing line while moving it.** Keeps click-click
  `armed` state intact on Esc/cancel and avoids drawing two lines from one
  blocker while reassigning it.
- **Multi-offer hand card drop label.** Shows "Choose how to play <Card>"
  rather than picking one offer's verb, opens the card menu at the drop rect,
  and restores focus to the hand card button on Cancel.
- **Scrolling battlefield rows.** A sweep counts only the visible portion of
  each card inside a scrolling row. In other rows it follows the card's drawn
  position, including an attacker leaning over the row edge.
- **Joint-combat mode across transient waits.** Keep `combatPreview` and the
  closed per-creature disclosure steady across `commandPending`, `work.running`,
  and `stopsPending` (disabling edits/confirm while `!live`), and preserve the
  draft when step 0 is rejected at the same revision.
- **Legal recipient outline during `connect` drags.** Outline exact legal
  permanent recipients (`connectTargets`, never other blockers) with dashed
  `--accent` while dragging a line, upgrading to solid `--accent` on snap.
