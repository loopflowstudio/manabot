# MTG-420 · Battlefield forms, copies and persistent ability sources

> “stop doing 1 card at a time”

Implement the remaining cards in batches that share rules behavior. Run the outer review cycle once per batch. Record which rows passed. When a row can't join its batch, record the specific dependency.

The Task brief's design constraint: “Explore one-card transform, two-card meld, and hidden face-down characteristics against contrasting real cards before fixing the representation, preserving exact physical-card identity and viewer-safe observations.”

## Done (detail in git and the test files)

- Transform: Cecil and Esper Origins, `fin.rs` (Cecil fixtures shipped before `9dbe17ee`).
- Meld: Fang/Vanille/Ragnarok with forked library order, Epic Downfall, lethal damage and Airbending Lesson, `fin.rs` at `e9b30cb6`.
- Face-down: Manifest Dread, Clone Legion copies, turn-up, sampling roots seeds 0–7, `dsk.rs` at `3672c184`. Academy recovery and manifest text at `fde87fb7` (`tests/academy/test_manifest_recovery.py`, `tests/semantic/test_manifest_text.py`).
- Departed Clone, Joo Dee and Homunculus Horde from the departed appearance, `lea.rs` at `bf9f4d73`.
- Copied triggers and scheduled sacrifice: Electroduplicate (flashback, under Doubling Season), Rite, Fire Crystal, Driver tokens, `fdn.rs` `copy_sacrifice` at `cc1061b7`. The stack keeps the triggering face (`238d8ecf`) and the captured catalog program (`320a8875`).
- `CopyException` restricted to copiable characteristics, `320a8875`.
- Named Koma's Coil copies through Rite, Self-Reflection (hand and flashback), Relm's Sketching and Extravagant Replication. Ordering Commands replay from the constructed snapshot. `fdn.rs` `named_coil` at `d52156c9`.
- Compress `9e297f0c`: waiting triggers read the captured face (`pending_trigger_definition`/`pending_trigger_face`), and target legality shares `fits_trigger_clause`. The focused nextest run passed 42 cases. The 13 Python copy-text, Academy trigger-order and pinned-game checks passed. The merge `039b275b` brought only Loopflow workflow files.

## Remaining batches

1. **Exact departures and linked returns (next).** Banishing Light, Stasis Snare and Prayer of Binding, then Kykar and Hoarding Dragon on the same exile, source-departure and reentry setup. Extend its graveyard and end-step branches to Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call, Fake Your Own Death, Undying Malice and Angelic Destiny.
2. **Graveyard choices, costs and simultaneous movement.** Use one two-graveyard setup for Raise the Past, Rise of the Dark Realms, Driver of the Dead (remaining target restrictions), Fiendish Panda, Heartfire Immolator, Sun-Blessed Healer, Nullpriest of Oblivion, Feldon's Cane (`atq.rs`) and Ordeal of Nylea. Abyssal Harvester is blocked (see below).
3. **Effective characteristics and grants.** Use one board before and after grant, copy, attachment movement and reentry for Celestial Armor, Dropkick Bomber, Kellan (`fdn.rs`), Mild-Mannered Librarian, Soulstone Sanctuary, Crawling Barrens, Witness Protection, Eaten by Piranhas and Imprisoned in the Moon (`emn.rs`).
4. **Representation review, consumers and budget.** See the sections below.

Row identity comes from the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) (Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md)). Update each row with its delivered source and checks. Reuse first printings. Match produced tokens by their full printed characteristics: Kykar needs a white flying Spirit, and the catalog already defines Treasure and Koma's Coil.

| Rows | Behavior to exercise |
| --- | --- |
| Banishing Light, Stasis Snare, Prayer of Binding | Remove the source before its entry trigger resolves, then again after the exile resolves. Only the second has an exile to end. Move the exiled card out and back before the source leaves. The new appearance stays put. Prayer gains life with zero targets chosen. A sole target that becomes illegal loses the entire resolution. |
| Kykar; Hoarding Dragon | Choose Kykar's blink or its Spirit. The delayed return binds to the object in exile. With two Dragons that found different artifacts, only the dying Dragon recovers its linked card. Bounce and replay the same Dragon, then check the old link. |
| Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call | Capture the Familiar's revival count at death. Spend one on its next-end-step return. In a sibling fork, move it out of the graveyard. Vessel and Call read death-time creature types. Counters and type grants attach to the returned incarnation. Kill Call together with an eligible creature. The trigger still returns the owner's card. |
| Fake Your Own Death, Undying Malice, Celestial Armor | Track which creature received a grant and when it ends. Move the Equipment after its entry trigger resolves. The protection follows its original recipient. When Fake Your Own Death is cast on a stolen creature, its owner gets the tapped card and the trigger's controller gets the Treasure. A dying token still produces Treasure. Malice returns the card tapped with its counter. |
| Angelic Destiny, Ordeal of Nylea | Follow the enchanted creature through death and the Aura through its own departure. Only the eligible Aura appearance is recovered. Resolve Ordeal's attack, counter threshold, sacrifice and optional library selection through its actual triggers. |
| Raise the Past, Rise of the Dark Realms, Driver, Fiendish Panda, Heartfire Immolator | Put qualifying and excluded cards in both graveyards. Check simultaneous entry, ownership, resulting control and ETB triggers. Driver and Panda choose legal graveyard targets when their triggers are placed, including Panda's last power and its exclusions. Pump Immolator, sacrifice it as the cost and return it before resolution. Its damage uses its departed power. |
| Sun-Blessed Healer, Nullpriest of Oblivion, Feldon's Cane | Read the actual kicked cast receipt on entry. An ordinary blink supplies none. Each has its own graveyard target restrictions. Check competing restricted spends on the kicked cards. Cane exiles itself as the cost before shuffling. |
| Characteristics subset | Kellan and Librarian: incarnation-bound upgrades. Sanctuary and Barrens: persistent versus temporary animation. Dropkick Bomber: flying and a damage trigger granted to one exact recipient. Witness Protection, Eaten by Piranhas and Moon: layered form changes across a copy. |

Pair Malice or Valkyrie's Call with simultaneous deaths and a blink in response. Keep simultaneous reanimation and sacrificed-source power in the integrated cases.

**Blocked: Abyssal Harvester.** It needs the history/queries Task's exact graveyard arrivals this turn from any zone. `TurnState::graveyard_from_battlefield` records only battlefield departures. The FDN ledger names no release that provides this. When the query lands, locate its consumer API with `rg -n 'graveyard_from_battlefield|GraveyardThisTurn|graveyard.*this turn' managym/src/flow managym/state/src`. Then offer Harvester a milled creature, a dead creature and a card that left and reentered. Follow the selected exile appearance through the Nightmare copy and the exile of the other Nightmare tokens.

## Open checks on finished work

- Lost Days: the order choice should be visible only to its owner, and the game should replay from its record.
- Horde, Driver and Fire Crystal still need viewer projections and replay of their recorded Commands from a recorded root. Their native fork and restore checks already pass.
- The named Coil setup has no seed-to-position replay file.
- Manifest sampling: there is no pending private selection in the fixture. Constrained sampling, paired seeded searches and comparative viewer trace exports remain open.
- Audit immediate battlefield move callers by their legal source zones. Start with linked entry follow-ups that consume a move's arrivals.

## Representation review

Review the final Rust representation against these witnesses.

- **Cecil** keeps its card, object, counters, attachments and targets across transformation.
- **Esper Origins** gets a new exact identity at each zone move.
- **Ragnarok** is one permanent backed by two physical cards, and each card moves exactly once.
- **A token copy of Ragnarok** moves no physical card.
- **A copy of a manifested permanent** gets the public 2/2 without the hidden printing or turn-up.

Extending the front-face field alone can express Cecil, but it drops a physical card when Ragnarok departs and gives a copy the hidden printing. The preferred shape is a battlefield form recording its backing card(s) and orientation, plus a separate immutable value for what an effect can copy. Keep the common single-card path compact. Tokens get their own origin, including a transforming token's faces when the creating effect copies them. Keep one permanent row, even for meld, and never fabricate a physical card. Review direction for meld: “In the FIN tests, fork and restore the pending library-order choice, take each legal order, and verify that each original card moves exactly once. Repeat with a token copy of Ragnarok and verify that it moves no physical cards.”

Native owners: `managym/state/src/{game_object,permanent,ability,effect}.rs`, `managym/src/flow/{identity,zones,layers,statics,trigger,triggers,proposed_event,resolution}.rs`.

- Keep `CardId`, `ObjectId`, `PermanentId` and viewer IDs distinct.
- Capture the departing form before `Object::mint_appearance`. Bind a link to the exact object its move produced.
- Keep `ExileLink`, `DelayedTrigger` and `AbilitySource` as separate types for their separate lifetimes.
- Preserve the two-arrival batch through replacement and rollback.
- `CopiableValues` feeds departed-source reads and token creation. The content pack decides when a departure needs a snapshot.
- Keep counters, animation, equipment and transient grants out of the copied payload.
- A stacked grant still resolves after its grantor leaves or the source changes face. A new appearance starts with only its own grants.
- Invalidate the layer, mana and trigger indexes on form, grant, control and zone changes.
- Planeswalkers, emblems and castable spell-copy links stay with their milestone owners.

After the cohort, review Clone, Electroduplicate, Harvester, Manifest Dread and Ragnarok together for shared movement, snapshot, ability and observation owners. Remove superseded fields and callers, and keep the directed regressions.

## Consumer checks

- One native viewer projection (`flow/knowledge.rs`, `agent/{observation,observation_json,observation_encoder,shared_effects,record_vocabulary}.rs`, `src/cards/tokens.rs`, bindings, `decision_context.rs`, `manabot/semantic/text_observation.py`) carries form, copy exceptions, grants, expiry and permitted knowledge. Legal actions, focus, history and relations use the same permanent reference. Hidden keys stay out of opponent joins, copy provenance, offer labels and tokens. Add the matching `observability.rs` entries. Preserve the v9/v10 layout and frozen evidence.
- Face-down comparison: build two roots that differ only in which hidden card a slot holds. At the opponent's decision, compare native and Python observations, v11 buffers, LLM text, Academy reconnect, viewer history and replay export. Check controller inspection separately, along with reveal on departure, `possible_worlds.rs` and search, and closed disclosure.
- Submit actual Commands at the private selection, kicked payment, meld order and trigger order. Fork while each is pending, reject the stale Command, and check that each consequence commits once. Replay each branch from its recorded root.
- For each group, retain the setup, seed, chooser, offered Commands, the branch taken, the outcome, viewer rows, text and tokens, the replay witness, the test path and the commit.

Focused checks from `managym/`: `cargo test -p managym-cards --test cards fdn::` and the matching set filters, `cargo test --test rules_tests cr_400`, `cargo test --test canonical_replay_tests`, the possible-world and search tests, and the learning-token test in `observability.rs`. After rebuilding the extension, run `uv run --extra dev pytest tests/llm/test_llm_native_game.py tests/env/test_observation.py tests/model/test_effect_vocabulary.py tests/model/test_pinned_pilot.py tests/academy/test_choice_navigation.py -q`. The full debug suite, format, clippy and package gates run at gate.

## Measurement

Add bounded states to `managym/examples/bench_states.rs`, varying the permanent count independently of source variety:

- Many Coils with Electroduplicate and kicked Rite, with variants that add distinct exceptions and Kellan grants.
- Many Banishing Lights with one source leaving. Mixed Familiar, Vessel and Call deaths. Healer kicked and unkicked.
- Several manifests over distinct printings, plus one Fang/Vanille pair and several namesakes.

Gate whole-game step throughput and Search-64 at the same-runner 10% limit. Attribute any digest change to a rules change ([engine budget](../docs/development.md#engine-performance-budget)). The 4096-byte inline `GameState` bound holds because trigger placement is boxed (`fde87fb7`). Its fork and throughput cost still needs measuring.
