# Assumptions made without an answer (MTG-344 kickoff, 2026-11-13)

Each is reversible before the PR lands.

1. **Python class name is `managym.Object`** (native `ObjectData`), matching the
   engine's `Object`. Alternative: `GameObject`, if `Object` reads too close to
   the builtin `object`.
2. **Removal release is 0.3.0**, the release the existing `learning_*` aliases
   name. If 0.3.0 is cut before anyone has run a release with these warnings,
   the aliases would warn for zero published releases; the alternative is 0.4.0.
3. **Encoder buffer struct fields keep the buffer-key spelling** (`agent_cards`
   ...), although the Task lists "their encoders' Rust names" in scope. Only
   functions and parameters that take the records are renamed.
4. **Text observation wording changes and `REPRESENTATION_VERSION` goes to
   `manabot-text-7`.** The Task puts the wording in scope; it does not mention
   the version. Alternative: leave the wording, and the version, alone.
5. **Event and action enum members keep `CARD`** (`EventEntityKindEnum.CARD`,
   `EventTypeEnum.CARD_MOVED`, `ActionEnum.SELECT_CARD`): `Object = 4` already
   exists and the names are in protocol fixtures.
6. **`StackTarget.card_id` keeps its name**: only graveyard-card targets write it.

7. The Python encoder already has a generic `_encode_objects` helper. The
   renamed record encoder is `_encode_object_rows`, using that existing helper.
8. Corrected prewritten docs: object rows include battlefield objects too
   (`add_permanent` pushes them), rather than only objects outside battlefield.

9. The requested full parity file hit the 300-second per-test limit in
   `test_observation_parity`, inside NumPy buffer comparison. Split its 200
   games into twenty ten-game batches with seeds 42–61, retaining every-buffer
   comparisons and the total game count. The first batch retains seed 42;
   later batches use independent seeds to stay within the mandated limit.
