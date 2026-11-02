//! Tests for cards filed in `src/cardsets/iko.rs`, in card order.

use crate::support::{
    game::{five_color_setup, until_end_of_turn},
    inspect::permanent_named,
    play::{
        activate_first, cast, choose, pass_until, play_and_settle, select_card, settle,
        target_opponent,
    },
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        card::Keywords, effect::Modification, game_object::PlayerId, permanent::CounterKind,
        zone::ZoneType,
    },
};

#[test]
fn death_rewards_and_selected_activation_costs() {
    let mut s = five_color_setup(&[
        "Bastion of Remembrance",
        "Callous Inspector",
        "Weaponize the Monsters",
    ]);
    play_and_settle(&mut s, "Bastion of Remembrance");
    assert_eq!(s.battlefield_permanents_named(0, "Human Soldier").len(), 1);
    let fodder = play_and_settle(&mut s, "Callous Inspector");
    play_and_settle(&mut s, "Weaponize the Monsters");
    let life = s.game().state.players[0].life;
    let weapon = permanent_named(&s, "Weaponize the Monsters");
    activate_first(&mut s, weapon);
    target_opponent(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(fodder),
        Some(ZoneType::Battlefield)
    );
    select_card(&mut s, fodder);
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
    let mut s = five_color_setup(&["Bastion of Remembrance", "Cycle of Renewal"]);
    play_and_settle(&mut s, "Bastion of Remembrance");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let source = s.game_mut().state.permanents[land].as_mut().unwrap();
    source.animated = true;
    source.change_counters(CounterKind::PlusOnePlusOne, 1);
    let card = source.card;
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select_card(&mut s, card);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 19);
}

#[test]
fn weaponize_keeps_its_exact_damage_source_after_removal() {
    let mut s = five_color_setup(&["Weaponize the Monsters", "Callous Inspector"]);
    let weapon = play_and_settle(&mut s, "Weaponize the Monsters");
    let fodder = play_and_settle(&mut s, "Callous Inspector");
    let source = permanent_named(&s, "Weaponize the Monsters");
    until_end_of_turn(
        &mut s,
        source,
        Modification::AddKeywords(Keywords {
            lifelink: true,
            ..Default::default()
        }),
    );
    activate_first(&mut s, source);
    target_opponent(&mut s);
    select_card(&mut s, fodder);
    // Model a completed removal response before the activated ability resolves.
    s.game_mut().move_card(weapon, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21); // Inspector -1, Weaponize lifelink +2.
    assert_eq!(s.game().state.players[1].life, 18);
    assert!(s.game().state.events.iter().any(|event|matches!(event,managym::flow::event::GameEvent::DamageDealt {source:Some(card),target:managym::flow::event::DamageTarget::Player(PlayerId(1)),amount:2} if *card==weapon)));
}
