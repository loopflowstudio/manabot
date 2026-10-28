# TLA Rust authoring decisions — 2026-10-28

- Jack Heart's Rust-first direction supersedes the JSON authority recorded at
  `f674bcd8f:scratch/questions.md`. Typed source programs are serialized offline
  into checked-in compiler inputs; JSON is generated output, never hand-authoring.
- Reversible choice: retain the established semantic compiler/lowering and exact
  serialized source identities. The Rust authoring types describe symbolic card
  programs; lowered IR owns numeric references and native runtime execution.
- Setup lists and independently reviewed qualification receipts remain JSON data.
  Runtime membership and fixed-token selection move alongside Rust definitions.
- `.lf/wave/engine/{GOAL,MEMORY}.md` are absent; no Wave writes are authorized.
- D1/f871a8 owns parser, construction and consumer input. Preserve its agreed pool API.
- Full TLA completion remains the accepted eight-PR plan. Existing setup availability
  never qualifies a whole card. No additional Task or merge is authorized.
- Integrated main includes MTG-177 repair; reassess affected qualification rather
  than retaining the obsolete claim that integration is pending.

Historical audit and MTG-207 notes were read and remain at
`f674bcd8f:scratch/questions.md`; they do not direct this implementation.
