# Assumptions and open items — MTG-389

Kickoff and concept review, 2026-11-18 (both headless; approved by nobody).
The brief asked to "stop for direction before building"; nobody was here, so
the plan orders the work so that the open choices are built last. The review
page is `scratch/concept-review.html`; its numbered questions match these.

Choices made for review:

1. **Analysis is made after the game, by re-executing it** (concept review).
   A played game already stores its seed, setups and every action, so the
   policy's reading, your view at every step, both hands and the game's
   numbers are derived from it. Passed over: the kickoff plan's advisor,
   loaded in your seat during every game and written into each step with
   your view and the spectator view. Why: it had to be chosen in Setup
   before a game for something wanted after it, could not analyse a game
   against Search or re-read an old game with a new checkpoint, and stored
   what the engine reproduces.
2. Derived views are not stored. A game recorded by another world or card
   set opens from its stored track as today, without the derived parts.
   Passed over: writing the derived views into the row at completion so
   they survive a world change (doubles the finished row).
3. The word is **Analysis** ("Analysis by <policy>", **Show analysis**,
   "analysis shown"), one word for the post-game reading and the live one.
   Passed over: "Advice"/"advised" (the kickoff plan's word; also the name
   of the unrelated fixture-bound `/api/advice`).
4. Analysis during play: allowed, off at the start of every game, turned on
   by the person, each decision made with it visible recorded and shown.
   Passed over: post-game only; always on; a separate practice game kind.
5. The default analysing policy is the opponent's own. Any registered
   policy can be asked afterwards.
6. Move marks state only what was measured (first choice ✓, or the policy's
   mass on your move). No chess.com-style grade names.
7. A completed shared game's analyses are visible to other readers.
8. No analysis by search: it plays out from the true hidden cards.
9. `/api/advice` is left as it is.

Shipping docs assumptions (2026-11-18):

- Use **Expected score** for `signed_outcome` and **Win chance** for
  `win_logit`. A signed outcome converted to 0–100 includes half the draw
  probability.
- **Show analysis** uses Review's eligible policy choices if the live opponent
  cannot analyse the game. If no policy is eligible, link to policy setup.
- The Games win rate uses wins divided by all completed games, including draws
  in the denominator, as in the existing session breakdown.

Left out, each wanting its own Task if wanted:

- Open hands during a live game (a native disclosure setting; it changes
  what the opponent's policy observes).
- Another line from a finished game after its live session is gone, and a
  live evidence provider for Study's Reveal. Both become a walk to decision
  N under this design.
- Card-level statistics from Academy games, and analysing a seat of a
  watched or imported game.
- Dropping the stored boards from a played game's row.

Not answered:

- **Whether a real played game re-executes to the same boards.** Read from
  the code (one mutation path, stored seed and action per step, setup
  rebuilt by `GameConfig.to_rust`), not run: this checkout has no built
  extension. Slice 1 is the test. If it fails for a reason that cannot be
  fixed, the kickoff plan's stored fields are the fallback.
- Time for one whole analysis with a current-world checkpoint on CPU: no
  w17 checkpoint exists on this machine. Slice 2 measures what is
  registered; the fixture policy proves the path, not the cost.
- Whether opponent moves can be fully named from the person's view
  (MTG-384 round 7 folded invisible steps into the move where they become
  visible; the same rule is assumed).
- Whether a policy session can be called only at one seat's decisions
  without having "observed" the steps between (the runner reads the
  observation and `capture_decision` at call time; manabot's session is
  assumed to need nothing else).

2026-11-18 Opus pass: the questions above are drawn as mockups in `scratch/explore.html`; awaiting answers before any slice.

2026-11-24, Jack Heart on question 1 (analysis by re-executing the game):
"1 is the right general thread. can we do a side by side with chess.com to see
what elements we might want to bring over". Side-by-side drawn in
`scratch/chesscom.html`.

2026-11-24, Jack Heart on the chess.com comparison: "I think my bigest POV
here is that the models are currently worse than me. So i think the approach
of showing how moves are rated visually and the data viz is helpful, but te
frame should not be that if i disagree its a mistake. its just taht
contextualizing what the agent things against what i actually did helps me
understand what the agent is doing". Review is for understanding the agent:
disagreements are neutral facts, never losses or mistakes.

2026-11-24, Jack Heart's answers to `scratch/chesscom.html`:
"1a yes / 2 yes / 3 sure / 4 off by default but available for sure".
- Each option shows the agent's probability and its estimate after the option.
- See its line is built in this Task.
- Store a time on every step.
- Analysis during play: available, off by default (marked when shown).

Slice 2 choices (2026-11-24, headless):

- Starting an analysis is participant-only; readers of a shared finished game
  see saved analyses but cannot start one (the plan allowed any reader). The
  record middleware already makes every mutation participant-only.
- Only your own seat is analysed. The opponent's seat (allowed under Show both
  hands) is not built.
- An undeclared value semantic draws no line and no option estimates (the plan
  drew raw values on their own scale); probabilities still show.
- Eligible policies: registered, admitted for the game's decks, with an
  artifact, not a Search composition. Scripted players are left out.
- A game still being played cannot be analysed yet (`analysis_game_active`);
  live analysis is slice 6.
- The analysis walk checks actor and action description per step, not whole
  boards (boards cost ~3 ms per projection; Review's views already check them).
- Pre-existing, seen in the proof: the status reads "Finished · You wins".

Slice 5 choices (2026-11-24, headless):

- The revealed hand opens from the plate's Hand count (the graveyard's zone
  dialog), not as face-up cards drawn in the top seat. Smaller and keyboard
  accessible; a drawn top-seat hand row would take board height.
- A branch is a note in the move list only, not a mark on the Strip or
  Transport. Only determinize, library reshuffle and reseed count as
  branching; set_decision_memory and force_action do not.
- The 69-decision Elemental snapshot from PR #577 was not found; constructed
  branched replays stand in.
- Analysis choices moved to their own read (`/analyses/policies`) rather than
  an in-process cache, so nothing can go stale.

Slice 3 choices (2026-11-24, headless):

- Damage needed a native count: nothing captured it, and presentation events
  miss the transition that ends a game. Added to the exposure capture.
  "Damage dealt" excludes damage a seat's sources dealt to itself; "taken"
  includes it (a random player bolting itself shows taken > opponent's dealt).
- "Spells cast" counts cards put on the stack, so copies are not counted.
- Turns are the game's turn number, as manabot's PlayCounts does, not turns
  each seat took.
- The record table groups by opponent and both decks together (one table),
  not separately by opponent and by matchup. Seat and disclosure are merged.
- Games has no agreement column: it would read every analysis's evidence per
  list row. Agreement shows in Review's Menu.
- The result card's numbers line appears once the server has counted the game
  (a fetch after game over); with no count it is omitted rather than parsed
  from the move list.
- Readers of a shared finished game may read its numbers.

Slice 4 choices (2026-11-24, headless):

- **See its line is a computed play-out, not a playable Study branch.** The
  policy's first choice, then the policy (first choices) in your seat and the
  game's own opponent (its own choices, seeded) in theirs, to the end of the
  next turn. "The person plays on" is not built: Study's Retry authority needs
  the live session's retained roots and plays one command. A playable branch
  from a re-executed root is the next step if wanted (a `StudyForkProvider`
  over `agent_line`'s walk, or a new Play game whose prefix is the record's).
- The line plays in the real game's world (not re-dealt): the game is over and
  Show both hands is already allowed. It can show cards the real game never
  drew. Opponent moves are named by their actions only for a complete game.
- The agent's choices in the line are its first choices (argmax), not samples,
  matching "its choice" in the analysis.
- Key moments: up to five differences (largest probability gap, ≥ 5 points)
  and three estimate movements (largest change by your next analysed decision,
  ≥ 5 points), in game order, computed in the client. Untrained policies
  produce few or none.
- In the throwaway proof the default analysis did not start: the first
  registered descriptor (Heuristic) was the opponent, not the Reader. Not a bug
  in this slice.

Slice 6 choices (2026-11-24, headless):

- "Shown" means the reading's message was sent to the pilot's socket before
  the commit. The server cannot see the screen. A reading sent while the
  page was hidden still counts.
- When the opponent cannot read the game, Show analysis picks the first
  eligible policy rather than asking first. The policy menu appears once
  the switch is on and more than one policy qualifies. With none, the switch
  stays off and says "No registered policy can read this game." with the
  managym.bots hint (there is no in-app policy setup page to link to).
- Watchers of a shared table never receive readings; the switch is the
  pilot's (`SUBMIT_LIVE_COMMAND`).
- The live analysis stays open after game over until the next game or
  session close; it reads nothing more.
- The folded Strip's live value points are not drawn. Probabilities and
  estimates are on the offers.
- `analysis_shown` is counted in `GameNumbers` (saved at completion), so
  Games and My record read it from `numbers_json` with no new column. A game
  whose numbers cannot be computed shows no count, though the events keep
  the mark.

Playing on and the acceptance spec (2026-11-25, headless):

- Playing on is a new Play game whose opening moves are the record's, not a
  `StudyForkProvider` over a walk: it reuses Play whole (opponent, Show
  analysis, recording, Review, numbers) and needs no retained roots.
- It starts at the decision itself; you choose there, including the policy's
  choice if you want it. There is no "start from its choice" button, and no
  mode where the policy keeps choosing in your seat (See its line shows that,
  read-only).
- Only your own finished games, at your deliberate decisions. Automated
  passes in the prefix are replayed as Commands (every window surfaces during
  the prefix), so the new game's early events are not marked `auto`.
- Prefix events get a new `committed_at` (the time they were replayed).
- A played-on game is left out of My record and marked in Games; its Review
  names the source game. Its numbers count the replayed moves too.
- The live move list of a played-on game starts at the decision, as a play
  demo's does; Review lists the whole game.
- The Play page asks with a notice and **Play on** button rather than
  starting on load, because the tab may be resuming another game. No
  confirm dialog; the notice says it stops the current game.
- The acceptance spec plays Bolt decks (Mountain/Lightning Bolt) against the
  Reader, not Lessons v Allies, so each game is a few turns. Its pick loop
  sometimes bolts its own face; the assertions do not depend on who wins.

Concept review of the built Study (2026-11-25, headless; `scratch/concept-review.html`):

- Try another line is drawn only while Study is available; the disabled
  button and its "retained rules state" caption go (page 1a). Passed over:
  removing Study's branch from played games (it has Reveal and Compare plans).
- Review draws no priority-stop toggles and no empty Stack block (2a).
- Review's rail height goes to Moves and the Analysis list (3a).
- Unchanged, open: Show both hands behind the Hand dialog (4a); "This game"
  in Review's Menu (5a).
