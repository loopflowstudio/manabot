# MTG-297 questions

- Trigger sources are read when the source is tapped for mana, not when the
  output enters the pool. CR 106.12a triggers on the production; nothing in
  current content changes the board between the tap and the production except
  the source's own sacrifice, which the earlier read preserves as LKI. If a
  mana-ability suffix ever removes or adds a triggered mana ability, move the
  read to production with source LKI. (Simpler choice taken.)
- A fixed (non-activated) mana source producing two types under Heartbeat
  would need a bonus type choice inside automatic payment; the planner leaves
  such a source out. No current fixed source produces two types.
