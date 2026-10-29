//! Ikoria: Lair of Behemoths (`src/cardsets/iko.rs`).
use crate::support::{
    activate_any, cast, choose, enter_or_cast, pass_until, permanent, select, settle, setup,
    target_opponent,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn death_rewards_and_selected_activation_costs() {
    let mut s = setup(&[
        "Bastion of Remembrance",
        "Callous Inspector",
        "Weaponize the Monsters",
    ]);
    enter_or_cast(&mut s, "Bastion of Remembrance");
    assert_eq!(s.battlefield_permanents_named(0, "Human Soldier").len(), 1);
    let fodder = enter_or_cast(&mut s, "Callous Inspector");
    enter_or_cast(&mut s, "Weaponize the Monsters");
    let life = s.game().state.players[0].life;
    let weapon = permanent(&s, "Weaponize the Monsters");
    activate_any(&mut s, weapon);
    target_opponent(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(fodder),
        Some(ZoneType::Battlefield)
    );
    select(&mut s, fodder);
    assert_eq!(
        s.game().state.zones.zone_of(fodder),
        Some(ZoneType::Graveyard)
    );
    settle(&mut s);
    assert_eq!(s.game().state.players[1].life, 17);
    assert_eq!(s.game().state.players[0].life, life);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
}

#[test]
fn bastion_uses_a_sacrificed_lands_last_known_creature_type() {
    let mut s = setup(&["Bastion of Remembrance", "Cycle of Renewal"]);
    enter_or_cast(&mut s, "Bastion of Remembrance");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let source = s.game_mut().state.permanents[land].as_mut().unwrap();
    source.animated = true;
    source.plus1_counters = 1;
    let card = source.card;
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select(&mut s, card);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 19);
}

#[test]
fn weaponize_keeps_its_exact_damage_source_after_removal() {
    let mut s = setup(&["Weaponize the Monsters", "Callous Inspector"]);
    let weapon = enter_or_cast(&mut s, "Weaponize the Monsters");
    let fodder = enter_or_cast(&mut s, "Callous Inspector");
    let source = permanent(&s, "Weaponize the Monsters");
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .temp_keywords
        .lifelink = true;
    activate_any(&mut s, source);
    target_opponent(&mut s);
    select(&mut s, fodder);
    // Model a completed removal response before the activated ability resolves.
    s.game_mut().move_card(weapon, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21); // Inspector -1, Weaponize lifelink +2.
    assert_eq!(s.game().state.players[1].life, 18);
    assert!(s.game().state.events.iter().any(|event|matches!(event,managym::flow::event::GameEvent::DamageDealt {source:Some(card),target:managym::flow::event::DamageTarget::Player(PlayerId(1)),amount:2} if *card==weapon)));
}
