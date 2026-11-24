No scratch design existed at the checkout base. The working plan uses the accepted Task 02 milestone and exact set-ledger selectors. Monica Rambeau // Photon, Living Light is the chosen modal casting probe. Its broader battlefield-form work stays with the forms task.

The first implementation slice uses Zul Ashur and a pair of Hungry Ghouls. This isolates exact-object permission and ordinary graveyard casting before adding Muldrotha's quota choices to the same position.

The compression pass covers the implemented Zul Ashur slice and its existing casting consumers. The later overlap fixtures cover multiple permission choices.

Concept review chooses direct native offers for consequential permission and profile choices. The alternative adds a permission-selection decision to search. Revisit against measured action width and search work.

Exact-object grants receive local keys when granted. Cleanup can reorder the surviving records without changing the meaning of a selected offer. An unambiguous priority action omits the route from its Command.

Singleton grants keep their existing wire shape for retained captures. The public Python list covers any number of grants. The cold-payment fixture uses a normally played Gray Ogre because a replay file needs a recoverable action history.

The next slice starts Smaug // Spew Flame in hand to exercise a second printed program before adding Bilbo’s competing exile replacement.

Concept review moves the Bilbo replacement fork ahead of batch authoring. Two successful casts ending in exile give a stronger test of grant ownership than comparing resolution with a counter alone.

The pending hand fixture uses twelve Mountains to cover both Spew Flame and the later Smaug cast in one turn. An opposing Counterspell branches from the same announced Adventure.

Use separate constructed and ordinary-play Smaug roots. The ordinary-play root supplies the recoverable history for cold replay.

The ordinary-play replay witness uses two Ogre targets and Lightning Bolt in each deck so the target choice and paid spell are both observable with native trivial skipping enabled.

Use Twinmaw's top-library versus hand offers as the first Omen witness. That comparison can expose a profile-filtering error before the shuffle implementation exists.

Use Pyromancer’s Goggles for the copied Omen witness. The copy can kill the sole target, so the physical card’s graveyard destination isolates the copy’s shuffle.

Use Scavenger Regent as the second Omen because its X cost exercises announcement and free casting through the same selected profile.

Keep the existing v11 stack mana-value column's printed-cost reading for saved models. Expose the native on-stack mana value separately and use Exude Toxin at two X values to distinguish the readings.

Exude Toxin’s cold replay uses Swamps against Mountains and Lightning Bolt. The opponent can respond after payment. The public stack record retains the number of printed X symbols so a captured observation can reconstruct the printed cost and effective value together.
Use Twinmaw and a second nonland card for the Etali fixture so the same suspended parent exercises profile, target and cast-order choices.

Use two prepared sources in the first priority fixture. A single Seed Suture cannot expose grouping by program name. Preserve physical-card Command bindings while carrying the exact offered object into announcement.

Use Regent’s existing Omen as the prepared-inset control. Changing only the inset kind isolates the hand and Cascade eligibility rule.

Use an unrelated exiled spell copy as the negative cleanup control in the two-source fixture.

For the two-copy fixture, both branches cast Counterspell through a native Command and then cast the surviving Seed Suture targeting the Angel. This checks that the surviving offer still executes after the counter.

The transfer review selects a fixed-player source-conditioned grant as the comparison. Transfer holds both object identities constant while distinguishing a change of caster from a loss of access.

The ordinary-play transfer replay uses Agent of Treachery to take the Angel and Spectral Sailor to keep a response available after payment.

Project the caster rule on visible permissions so a policy can distinguish the effects of a control change before choosing it. Use the existing transfer fixture as the paired control.

Use a castable Lightning Bolt to issue a public target address for the Angel before it leaves. Use a controlled non-Pilot creature as the positive target for All Aboard.

Use the pending Waypoint activation as the first fork for repeat preparation. Compare exact links and copy movements before adding a stored reverse index.

Use an ordinary-play Voxmancer alongside the constructed Angel for the Waypoint replay. Its upkeep can renew its spell between activations.

Use blink for the Emeritus source-loss branch. Resolving the original trigger with the same card back in play distinguishes source appearance from physical card identity.

Group the outstanding rows by preparation, printed spell programs, and zone permissions. Each group shares the native fixtures listed in the plan.

The six history-dependent preparation rows wait on MTG-418 instead of adding turn-history counters here. The brief says to defer such cards. Leech Collector counts as one: "first life gain this turn" differs from a once-each-turn trigger when life was gained before it entered.

Concept review (whole casting/permission area) makes four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Replace branch-added `action::{ADVENTURE, OMEN}` flags with one `action::SPELL_PROFILE` category (size 4: Front, Adventure, Omen, BackFace); zero pinned-pilot break because columns 25-29 are new on this branch.
- Model Omniscience as `PlayRoute::FreeFromHand` (deduplicated across copies) rather than overloading `kicked: bool`, so free casts still allow Kicker/Bargain (CR 118.9d) and compose with Harbinger's `AdditionalCost::FlashSurcharge`.
- Store exiled-card counters (Tinybones stash counters) in sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` instead of adding a `Counters` field to every `Object`, keeping normal object size/clone cost unchanged and preserving CR 400.7 staleness.
- Reuse `CardDefinition.back_face` with `modal_back: bool` and `SpellProfile::BackFace` (resolving with `ZoneEntry { transformed: true }`) rather than adding a fourth `InsetSpell::ModalBack` variant.

Preserve the compact priority projection's existing front-face AnyTarget ordering. Recorded Command preconditions use its positional candidate addresses. The complete compound API binds targets from its own action, including the selected face and payment route.

Use the original M11, M13 and Time Spiral printing keys for the Foundations reprints. Their game fixtures stay with the Foundations permission cohort.

Keep the prior opening authority and parity receipts by recording the casting catalog in new files. Their definition ordinals come from the catalog at `d12eb496b3`. Comparing the retained and current Commands by printed name also resolves the source of each CARD_REVEALED event.

Use Heartwood Crafter and Codie for the next casting-provenance slice. An exile cast of a linked prepared spell exercises both the spending restriction and Codie's copy. Use the existing Burst Lightning fixture for manual kicker, with one red unit spent on the generic cost.

Post-build concept review (`a5c3c68beb` vs `82c5d1d1b7`) records five reversible choices in `scratch/concept-review.html` and `scratch/plan.md`:
- Unify priority and resolution `CastChoices` construction behind one shared builder where `resolving: true` for ALL `CastDuringResolution` casts (free and paid) and `free: true` strictly means "without paying its mana cost" (fixing Harbinger of the Tides under Yue/Etali/Cascade and giving `from_prepared` one owner).
- Keep single-route `Action.route = None` on the wire for retained Command compatibility, and add `Game::resolved_play_route` so solitary `GraveyardType` (Muldrotha) plays always project `quota_type`, Muldrotha focus, and route description.
- Decouple `PlayGrant.face_down` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`, replacing the 5 visibility calls to `is_foretold` with `is_face_down_in_exile` / `can_view_face_down_exile` while keeping `is_foretold` strictly for Foretell cost and determinization (unblocking Gandalf's Flameshape).
- Publish Tinybones's static exile-counter permission on `LayerBoard.play_permissions.exile_counter_plays` as `PlayRoute::ExileCounter(CounterKind)`, backed by sparse `GameState.object_counters`.
- Keep `Action::SelectCard` for front-face `CastDuringResolution` offers and `Action::CastSpell` for non-front profiles to preserve retained replay captures.

## 2026-11-23 implement pass

- Codie's type line is "Artifact Creature — Book Construct". `Book` is filed as an artifact subtype in `state/src/subtype.rs`. That is an assumption: if the CR lists Book as a creature type, move it (one line).
- Stack spell rows (`SpellData`) carry neither `kicked`, `flashback` nor the new `from_prepared`. Codie's trigger reads `from_prepared`, so the policy cannot see whether a spell on the stack is prepared. That gap predates this pass; it is left for a shared observation change rather than one column.
