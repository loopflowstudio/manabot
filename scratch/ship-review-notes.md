# MTG-483 Ship Review Notes (2026-11-25)

## Verbatim words from Jack Heart

1. On how callers use grouped decisions:
   > so do you start managym with some sort of grouped decisions mode and you get a different api
   > or do you have to identify spots where grouped decisions are possible and submit them instead of a normal decison

2. On the PR #623 `group()` / `preview_group()` / `apply_group()` + `payload["group"]` summary:
   > This reads as kidna awkward to me

3. On `compound_offers` / `step_structured` in `managym/src/agent/structured_offer.rs`:
   > Is this actually used

4. On which part feels most wrong (Python/`managym` API vs Academy protocol split vs both):
   > mostly the former but its hard for the latter to be right if the former isnot

5. On `env.step(plan)`:
   > step taking a plan is amniguous to me - does it execute all the plan or just one part

6. On `env.decision()` mid-combat:
   > what does thisnreturn if youre half way throufh stepping through attackers

7. Outcome:
   > ok i guess lets build it

## Reading (separate from verbatim quotes)

- Send MTG-483 back to `build` for Pass 4 before delivery.
- Replace the side-channel `group(env, viewer=...)` / `preview_group(env, viewer, revision, fingerprint, plan)` / `apply_group(...)` API with a first-class unified decision API on `Env` (`env.decision()`, `env.preview(choice, *, at=None)`, `env.submit(choice, *, at=None)`) while keeping `env.step(action: int)` strictly for one flat sequential RL step.
- Deprecate the unused `compound_offers` / `compound_commands_json` / `structured_offers` / `step_structured` methods for `0.3.0` (patch-clean `main`) so `managym` has one grouped-decision surface.
- Align Academy's server/protocol cleanly with the new `managym` decision/choice model (while keeping `protocol/experience-v1.schema.json` frozen for replay `frame_hash` parity) and fix the two CI failures on PR #623 (`test_group_command.py:226` 405 vs 404, and `e2e/fixtures/combat.ts:319` `showEachCreaturesChoice` for library groups).
- No follow-up Tasks filed yet; keep them for the next `ship-review` after Pass 4.
