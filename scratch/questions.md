# MTG-154 open questions / assumptions

1. **Item 2 vs item 3 branch.** Real authored-game decisions (96 cases, two life
   levels each) rarely stick on main (see notes.md), so the additive saddle is
   mostly a property of the probe's minimal one-column contrasts. Assumption:
   FiLM in the gate (protocol learnability-2). Superseded in scope by item 3
   (FiLM for new runs). The gate hard-codes `policy_film=True` in
   `capacity.py`; the protocol version records it (no `Config` knob).
2. **Held-out Learn miss (~10%/seed, unchanged by FiLM).** Held-out variant 1's
   native shuffle puts definition 3 in hand; Learn fit cases never show it.
   Swapping it for a fit-seen definition fixes every checked miss. Not a stuck
   start; left unchanged because fixing it changes fit/held-out case design.
   With three seeds the strict gate still has ~25% odds of failing on an
   init-shifting change for this reason. Needs Jack Heart's call.
3. **Scope of "default for new runs" (Jack Heart steer, 2026-10-20).** Applied
   to `Hypers()` (the `manabot train` CLI and its presets, via
   `new_agent_spec()`) and the ordinary baseline recipe, bumped to
   `baseline-v2`. Left additive: `AgentSpec()` itself (omitted field =
   additive for frozen recipes/checkpoints), `ataraxos-mtg-v1` (pinned JSON),
   experiment runners' frozen regimes, and the `agent_hypers or AgentSpec()`
   fallbacks in `sim/distill.py`, `sim/value.py`, `sim/search_supervised.py`,
   which reproduce dated research, and `verify.util.build_hypers` (pinned
   additive: it feeds versioned first-light / exp-11 recipes). Reversible if
   those should follow too.
4. **CLI compound runs now need `agent.policy_film=false`.** The `Hypers()`
   default is FiLM, and compound + FiLM is rejected; the error names the fix.
   Alternative (not taken): silently drop FiLM when compound is requested.
