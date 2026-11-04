# MTG-242 decisions — 2026-11-04

Jack Heart authorized one engine change and a reviewable PR, stopping at review;
no merge or training run. The implement step owns code and focused checks only.

The inherited T7 note does not select this work. Its complete text remains at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.

Material assumption: the directive is the implementation plan; no separate
accepted design was supplied. The supported domain and remaining work are in
[mtg-242-mana.md](mtg-242-mana.md). Frozen policy inputs retain their meanings;
new restriction distinctions need an explicit input version, not wider old rows.

## Main sync — 2026-11-04

Preserve MTG-242's scope while integrating main's selected-cost X repair,
configurable native buffer axes and explicit first-legal action policy. V8
relation-width selection and mana spending contexts compose with those changes.
Incoming MTG-213 decisions remain historical context at
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`;
they do not authorize a training run or expand this Task's review boundary.
