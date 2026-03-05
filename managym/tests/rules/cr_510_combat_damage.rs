use managym::flow::turn::StepKind;

use super::helpers::*;

/// CR 510.1c — Unblocked attackers deal combat damage to the defending player.
#[test]
fn cr_510_1c_unblocked_attacker_deals_player_damage() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 101);
    s.setup_attacker_ready();
    s.declare_attack();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 1);
}

/// CR 510.1a — A blocked attacker assigns damage to creatures, not the defending player.
#[test]
fn cr_510_1a_blocked_attacker_does_not_damage_player() {
    let mut s = Scenario::new(forest_elves_deck(), forest_elves_deck(), 102);
    s.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    s.advance_to_active_step(1, StepKind::Main);
    s.play_land_and_cast_creature(1, "Forest", "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before);
}

/// Negative: if no attacker is declared, no combat damage is dealt to players.
#[test]
fn cr_510_negative_no_attacker_means_no_player_damage() {
    let mut s = Scenario::new(forest_elves_deck(), forest_deck(), 101);
    s.setup_attacker_ready();
    s.decline_attack();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before);
}
