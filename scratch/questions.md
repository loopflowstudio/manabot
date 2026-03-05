# Open questions / assumptions

- Stage 04 design assumed stage 02+03 were already landed (event queue + choose-target infra), but this branch did not yet include them. I implemented a minimal in-engine version of those prerequisites (runtime `GameEvent` queue, `ChooseTarget` action space/action, `Target` enum, and `StackObject` model) scoped only to what Man-o'-War ETB triggers require.
