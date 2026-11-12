//! Tests for cards filed in `src/iko.rs`, in card order.

use crate::support::{
    game::{animate, five_color_setup, until_end_of_turn},
    inspect::permanent_named,
    play::{
        activate_first, cast, choose, pass_until, play_and_settle, select_card, settle,
        target_opponent,
    },
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        card::{Keyword, Keywords},
        effect::Modification,
        game_object::PlayerId,
        permanent::CounterKind,
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
    animate(&mut s, land);
    let source = s.game_mut().state.permanents[land].as_mut().unwrap();
    source.change_counters(CounterKind::PlusOnePlusOne, 1);
    let object = source.object;
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select_card(&mut s, object);
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
        Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink])),
    );
    activate_first(&mut s, source);
    target_opponent(&mut s);
    select_card(&mut s, fodder);
    // Model a completed removal response before the activated ability resolves.
    s.game_mut().move_object(weapon, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21); // Inspector -1, Weaponize lifelink +2.
    assert_eq!(s.game().state.players[1].life, 18);
    assert!(s.game().state.events.iter().any(|event|matches!(event,managym::flow::event::GameEvent::DamageDealt {source:Some(object),target:managym::flow::event::DamageTarget::Player(PlayerId(1)),amount:2} if *object==weapon)));
}

#[test]
fn drannith_magistrate_stops_opponents_casting_from_the_graveyard() {
    use crate::support::{game::plant_in_graveyard, inspect::can_cast, play::castable};
    let mut s = five_color_setup(&["Drannith Magistrate", "Join the Dance"]);
    let dance = plant_in_graveyard(&mut s, 0, "Join the Dance");
    assert!(castable(&mut s, dance));
    // Your own Magistrate does not restrict you.
    let mine = s.force_permanent_on_battlefield(0, "Drannith Magistrate");
    assert!(castable(&mut s, dance));
    let theirs = s.force_permanent_on_battlefield(1, "Drannith Magistrate");
    assert!(!castable(&mut s, dance));
    // The restriction ends with the source.
    for magistrate in [mine, theirs] {
        let object = s.game().state.permanents[magistrate]
            .as_ref()
            .unwrap()
            .object;
        s.game_mut()
            .move_object(object, managym::state::zone::ZoneType::Graveyard);
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, dance));
}
