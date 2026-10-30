# PR 1: one TLA content authority and straightforward cards

Accepted by Jack Heart on 2026-10-28 as the first of seven TLA PRs plus one separate decklist-import PR (eight total). Automatically implement, test, refine and publish this one cohesive PR, then stop for Jack Heart's review. No auto-merge, land command, paid compute, training or follow-up Task launches. Source exploration: loopflow://manabot/task/0d9ba3. This document is the implementation brief; copied exploration files are supporting evidence and do not override this authorization.

Accepted revision, Jack Heart, 2026-10-28: typed Rust set files are the authored
authority; generated representations are checked in. See
[tla-rust-authoring-direction.md](tla-rust-authoring-direction.md). This supersedes
the earlier JSON authoring implementation, while preserving the compiler/runtime
and D1 ownership boundaries. Jack Heart's 2026-10-29 direction replaces the
qualification model below (support = defined in a set file; one world pool of
every defined card; set filters for TLA decks; every card in its first-printing
set file). Where this brief says "qualified", "eligibility" or "86 cards", read
the current plan in [892c47-author-tla-cards.md](892c47-author-tla-cards.md).

Outcome: current TLA behavior has one typed authoring source lowered to the native rules runtime and learning projection; a new card no longer needs an unrelated handwritten native registration just to compile or obtain display text. Extend this source with the missing cards whose complete behavior fits established primitives. Preserve the authored setups and existing behavior witnesses. Registration, native/compiled agreement and generated tests alone never certify a card's correctness.

The initial content cutover at `2a44b763c` removed duplicate native TLA registrations from tla.rs, elemental.rs, sacrifice.rs and temur.rs. The accepted Rust revision restores tla.rs as the single typed authoring source, replacing the intermediate JSON authority.  Avatar Enthusiasts and Glider Kids have typed programs and independent witnesses. Non-TLA behavior and token closure remain. One runtime executes reviewed programs; no browser rules, runtime Oracle parser, fallback registry or parallel current-world executors.

Final research from 0d9ba3 is available under `scratch/reference/full-tla-final/`; the earlier incomplete snapshot is historical. All 286 cards, 294 faces and token obligations are assigned in `docs/rules/tla/allocation.jsonl`. Five demonstrated prerequisite corrections leave 91 T1 owners. The T1 clause review and qualification records that followed were retired by Jack Heart's 2026-10-29 direction (history at `be2b7586b`). The allocation is reviewed, not fully delivered, and preservation never qualifies an incomplete card. Later mechanic-heavy cards remain unsupported and owned by T2–T7. The reconciled grouping is in `scratch/tla-delivery-batches.md` and the durable delivery-plan.json. Full-file acceptance is unchanged.

Parallel decklist Task f871a8 owns parsing, normalized submitted deck models, native construction/format admission and Academy/training input plumbing. This Task owns content definitions, eligibility and content versioning. The agreed Rust `content_pool(&str) -> Result<ContentPool, ContentError>` and `pool_content(&str) -> Result<Arc<ContentPack>, ContentError>` are exported from `cards` (`cards/pool.rs`); Python exposes `managym.content_pool(pack_key)`. The descriptor includes pack key, world version and content digest, and per definition its ID/key, Scryfall id, registry name, eligibility and reason (compiled-semantics hashes were removed 2026-10-29). `"world"` (`world::WORLD_POOL`, formerly `tla-pool-v1`) is the default consumer pool and describes the default pack; `PoolDefinition.card_def_id` is required and rows use `key`. The importer confirmed that novel combinations use its `admit_deck` before `pool_content`; existing exact authored presets remain available independently of pool qualification. Latest compressed integration and joined consumer validation remain D1 work.

Main's accepted MTG-177 resolution repair (`7d7736c6a`) is integrated in this
checkout. Current content is world w17 with learning schema 9 (one representation,
2026-10-29); historical world/content/checkpoint evidence stays frozen.

Open PR overlap checked 2026-10-28: PR383 (MTG-175 adaptive curriculum) touches vector_env.rs and python/vector_env_bindings.rs and reserves curriculum choices. PR389 (Task 19d240 shared evaluation) touches manabot/semantic/policy.py, model/semantic_cards.py and simulation consumers. Preserve their decisions and recheck status before touching those paths. No open PR was found changing the core cardset/semantic-content files.

Acceptance: existing admitted behavior survives the cutover; at least one genuinely new complete card compiles and executes without a handwritten registry entry; independent expected rules outcomes cover every newly admitted clause; incomplete definitions reject admission explicitly; native/learning/Academy consumers reference the same programs and identities. Exercise Command, replay/fork, observations and current-world admission where affected. Maintain uncapped choices and viewer safety. Demonstrate the production path, not only data generation.

The current evidence and remaining gate work are recorded once in `scratch/892c47-author-tla-cards.md`. Final review includes the admitted card list, remaining batches, behavior proofs and compatibility impact. The named two-player product format remains a D1/full-file review decision; content qualification alone establishes no tournament legality.
