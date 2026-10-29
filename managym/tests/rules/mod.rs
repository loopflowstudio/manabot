pub mod cr_103_game_start;
pub mod cr_106_mana;
pub mod cr_117_priority;
pub mod cr_305_lands;
pub mod cr_400_identity;
pub mod cr_405_stack;
pub mod cr_508_attackers;
pub mod cr_509_blockers;
pub mod cr_510_combat_damage;
pub mod cr_514_cleanup;
pub mod cr_601_casting;
pub mod cr_602_activated_abilities;
pub mod cr_603_triggers;
pub mod cr_608_resolution;
pub mod cr_614_replacement;
pub mod cr_702_keywords;
pub mod cr_704_sba;
pub mod event_observations;
// Shared with the per-card tests; this crate uses only part of it.
#[allow(dead_code)]
pub mod helpers;
pub mod presentation_events;
pub mod structured_attacker_offers;
pub mod structured_offers;
// Shared with the per-card tests; this crate uses only part of it.
#[allow(dead_code)]
#[path = "../cards/support.rs"]
pub mod support;
pub mod tla_mechanics;
