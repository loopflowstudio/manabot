# Marvel Super Heroes (MSH): complete-set planning

Research date: 2026-11-04. Priority: survives the next rotation. Reviewed all
281 cached Oracle identities, including all ten faces of five modal double-faced
cards (286 face records overall). The companion [ledger](msh.jsonl) has one
exact-ID/name row per input. It assigns integration ownership, not implementation
coverage. `high` means a straightforward definition obligation; `medium` marks
extension/composition work needing directed validation. Neither certifies runtime
correctness. No games, benchmarks or runtime changes were made.

The [June release notes](https://magic.wizards.com/en/news/feature/marvel-super-heroes-release-notes)
distinguish Standard MSH from MSC/MAR products and confirm direct casting of
either modal face. Teamwork status survives stack copying; power-up discounts
use mana cost rather than mana value. Harnessed status is noncopiable. The
[June update bulletin](https://magic.wizards.com/en/news/announcements/marvel-super-heroes-update-bulletin)
adds power-up/teamwork rules, heal, worthy, and clarifies repeated optional
payments producing reflexive triggers. These authoritative clarifications
resolve material ambiguities in the cached text. Current legality/bans remain
the parent inventory's separate responsibility; Commander examples in those
notes are not additional MSH ledger entries.

## Existing foundation and genuine gaps

There is no `cards/src/msh.rs`. Existing `ths.rs` owns Lightning Strike and
`lea.rs` owns basics; other reprints need first-printing placement. Native
selectors, `TargetClause`, exact `ObjectRef`, `OptionalPayment`, delayed triggers,
Sagas, crew, cast permissions, copy exceptions and layered characteristics provide
substantial reuse. TLA's prowess/exhaust, waterbend payments and targeted
continuations are especially useful predecessors, not exact replacements for
MSH rules.

Source inspection of `state/src/card.rs`, `ability.rs`, `mana.rs`, `effect.rs`
and `permanent.rs` confirms missing vocabulary: teamwork, power-up, connive and
improvise have no direct rule owners; spending predicates cover only
Lesson/Shrine/Ally; counter kinds omit shield, poison and most keyword counters.
The current transformed-card mana-value reader assumes the front face, unsuitable
for MSH modal faces. Duration variants omit several required next-turn and
source-presence lifetimes. The [shared engine audit](../current-engine-audit.md)
supplies the deeper replacement, resumable-iteration and LKI remediation; this
set should consume that work rather than add parallel handlers.

## Candidate batches and ownership

Batch numbers identify shared responsibility, not a rigid sequence. Build the
event, choice and payment foundations together, then layer power-up, copies and
replacement-heavy definitions; close exceptional setup/player control last.
Ledger dependencies name additional batches, including cross-cutting prerequisites.

| Batch | Cards | Authority and required integration |
|---|---:|---|
| MSH-01 definitions | 61 | Existing verbs, triggers, predicates and canonical tokens; ordinary Equipment, damage, buffs and reprints. |
| MSH-02 events/history | 43 | `GameEvent`/trigger matching owns simultaneous counter/tap/entry groups, source-specific and global turn history, and Plan thresholds. |
| MSH-03 costs | 44 | Casting/activation announcement and mana witness owner: teamwork, improvise, alternative costs, restricted mana and repeated/reflexive payments. |
| MSH-04 power-up | 23 | Extend activation accounting beyond exhaust's boolean: counts per exact ability/object, entry-turn colored/hybrid reduction and dynamic permitted count. |
| MSH-05 choices/zones | 66 | Resumable selection, scoped results, connive, multi-mode/target clauses, private choices, cast permissions, copying stack objects and exact linked movement. |
| MSH-06 characteristics | 26 | One layered view for copy exceptions, grants, CDAs, history-dependent traits, harnessed designation and duration/expiry. |
| MSH-07 replacements | 10 | Proposal state machine owns shields, counter modification, prevention/doubling, healing, finality and poison consequences. |
| MSH-08 faces/time | 6 | Modal casting plus transformation and scheduled turns/combats with attached restrictions or permissions. |
| MSH-09 setup/control | 2 | Pregame Quicksilver choice and Cosmic Cube's next-turn decision controller, including private information entitlement. |

Teamwork must be announced before modes and targets: Murdock's Crusade changes
mode count, while Cruel Alliance changes target eligibility. Reuse crew's
power-sum selection and waterbend's reservation infrastructure, but preserve
distinct tap provenance for Maria Hill. Improvise only pays generic spell costs.
Ronin requires an equip-ability predicate, whereas Hydraulic Helper allows
nonspell payments. Replace the fixed spending mask; appending Hero/Villain bits
would retain the wrong abstraction.

Power-up reuses exhaust's exact-object accounting but needs counted uses because
Wonder Man changes capacity and Kang prohibits activation during a scheduled
turn. Wonder Man leaving and returning must not refresh consumed uses. Plans
are ordinary enchantments with counters and triggers, not a new quest executor.
Keep trigger-frequency limits separate from optional once-per-turn actions:
Crossbones and Baron Strucker deliberately differ. Thor Odinson requires two
prowess occurrences, never a deduplicated keyword flag.

Retire the non-suspending iteration shortcut for Worlds Within Worlds and
multi-card casting; retain private selections before simultaneous entry. Replace
hard-coded discard/search composites with scoped selected-object results.
Secret Invasion, Absorbing Man and Taskmaster require temporary copies with
different exceptions and durations. Preserve the shared copy/layer owner rather
than attaching bespoke copied-stat fields. Wolverine and Leader require genuine
replacement processing, not after-event triggers.

## Objects, information and acceptance

Token closure includes Clue, Food, Treasure, Soldier, vigilant Hero, Wall,
hexproof Leviathan, Merfolk, menace Villain, 2/2 Robot Villain, named 3/3 Doombot,
Insect, Squirrel and mandatory-attacker Alien; plus legendary Void, Redwing,
Zabu, Tiger God, Galactus and Sturdy Shield Equipment. Preserve each token's
colors, subtypes, abilities and tapped-entry instruction. Copied tokens require
independent copiable values. Vibranium tokens belong to the excluded Commander
material, despite appearing in the shared notes.

Directed witnesses should include Maria Hill paying teamwork beside Captain
America's first-tap untap; Plan counter addition crossing its threshold under
Doc Samson; illegal optional power-up target versus selecting no target;
Wonder Man capacity after blink; connive after source departure under Leader;
shield/prevention/doubling choices; sequential versus simultaneous Wolverine
damage; Nick Fury entry then optional transformation; and Worlds Within Worlds
with both players selecting privately. Test Cosmic Cube with a nested payment
choice and a fork, verifying actor entitlement without leaking either hand.

Every new counter, history fact, permission, face choice and continuation needs
native/Python tensor parity, learning tokens, LLM text, semantic capture and
Academy's shared legal actions. Review tap/counter/hand-size cache invalidation;
preserve frozen input versions and fail explicitly on capacity excess. Use
Commands for suspended choices and compare clone witnesses, stale rejection and
replay identities.

## Runtime and search evidence

Measure ordinary `Env::step` and search reply cost separately on bounded
representative fixtures: teamwork subsets, mixed restricted mana, repeated
connive, several Plan triggers, temporary copies and two private selection
continuations. Avoid eager subset expansion and cloned effect trees; preserve
all legal choices through shared structured decisions. Keep the no-replacement
path cheap and index event/replacement sources by real dependencies. Retain
debug cache oracles and Engine Budget exact-result digests with same-runner
paired comparisons and cumulative milestone accounting. This research supplies
no speed claim, benchmark result or proof of every cross-set interaction.
