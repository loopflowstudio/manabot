# MTG-272 implementation — 2026-11-11

Ryan's request: "one more request: can you add true hands in batched mode?"
The originating clarification: "yeah, he is training a belief model."
Task scope also includes a code-evidenced scalar/batch API audit. MTG-271 owns
version/dependency/release files; no publication belongs to this step.

## Plan and reversible choices

Implement the contract in `docs/engine.md`, Choosing a Python API and Belief
hand targets: explicit native seats, complete name counts, detached typed
records, slot/game/seed/revision identity, native/CapturedVector/Torch access.
No observation, policy input, capture payload or frozen identity changes.

Use the existing private Env reset generation as the game label identity.
Allocate its values process-wide instead of per Env so replacing a slot's Env
or rebuilding the Torch wrapper's native vector cannot reuse a generation.
It is an opaque lifetime token, never a replay/world identity or promised
consecutive counter. Capture origin and stale checks retain their semantics;
the token is not serialized into public captures. No hand work occurs until
requested. Failed-call rows remain caller-discarded as documented.

Delete — do not maintain: no predecessor API is replaced. Drop inherited
b015d5/C2 scratch, retained at
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md` and its linked
artifact evidence. Those experiments and their open decisions are unrelated.

Remaining: implement the accessor and typed wrappers; audit actual bindings,
wrappers and installed use; add bounded parity/privacy/reset/mask tests;
focused debug check, atomic release extension rebuild and Python proof.
Gate owns affected suites and broader acceptance checks.
