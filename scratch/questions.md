# MTG-286 decisions and evidence — 2026-11-11

Jack Heart: "lets file a task for covering all of TLE".
Inherited MTG-285 release notes remain in `scratch/questions.md` at checkout
base `4c041d6cefe829c5b1b52273ba19991d663ee26b`; they do not direct this Task.

The proposed design and open rules questions live in
`scratch/mtg-286-define-every-tle-card-so-tla-limited-pools.md`; the dated
Scryfall inventory lives in `scratch/tle-inventory.md`. No accepted design was
supplied, and no runtime implementation or corpus admission proof has run.
The shipping guide at `docs/engine.md#tla-decks-with-tle-bonus-sheet-cards`
still describes the intended outcome.

Reversible assumptions: extend the existing deck-import guide; retain the
current 1v1 development world with correct no-commander behavior. Full Commander
setup/multiplayer remains outside the proposal without narrowing the TLE list.

The implementation step stopped for rules-design review under its instruction:
"If implementation turns up a counterexample that breaks the slice, the data
model, the deletion path or the overall design, stop dependent work and revise
the design or send it back to review." It made no Task/Graph transition.
The plan records the payment-model counterexample and other semantic gaps.

Earlier checks: the Python example's syntax, paired fences, Academy link and
`git diff --check` passed; its APIs were checked against stubs and bindings,
not executed against completed TLE definitions. `lf open --host desktop --file
docs/engine.md` reported "No Desktop Files view attached; nothing opened."
The inventory matched all 242 fetched names in order, with 234 missing literal
definitions. The corpus was absent locally but present in the artifact manifest;
fetch through Doppler when admission validation is reached.

Compression keeps rules findings in the plan and evidence here, removing
repeated status and repository-wide instructions. The external inventory is
unchanged. Original uncommitted notes were moved before editing to
`/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg286-compress-ju2m728q/`.

Check — `uv run --no-project python` (inline syntax/fence/link/inventory check)
and `git diff --check` passed. All 242 inventory rows match the retained context
snapshot; 234 remain missing. No runtime behavior changed or suite was rerun.
`lf context` remains within both scratch limits (12,000 tokens / 98,304 bytes).
