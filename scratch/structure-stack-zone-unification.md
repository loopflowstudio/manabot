# 02: Stack/Zone Unification — Validation

## Verify

```bash
# All stack mutations go through coordination methods
grep -rn 'state\.stack\.push\|state\.stack\.pop' managym/src/flow/
# Expected: only zones.rs (the coordination methods)

# Debug invariant fires after every step()
grep -rn 'assert_stack_consistent' managym/src/flow/tick.rs
# Expected: called in step()

# Tests pass
cd managym && cargo test
```
