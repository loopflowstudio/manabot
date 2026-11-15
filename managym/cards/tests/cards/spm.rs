//! Supportive Parents consumes shared taps without summoning-sickness or
//! mana-tap semantics.
use crate::support::inspect::zone_of;
use crate::support::{
    game::{five_color_unfunded_setup, move_to_hand},
    play::command,
};
use managym::{
    agent::{
        action::Action,
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    state::{game_object::PlayerId, mana::Mana, zone::ZoneType},
};

/// Two activations pay one spell from four distinct creatures. Each selected
/// tap commits before the output exists, the continuation states the unpaid
/// count, and selecting a creature for the cost is not tapping it for mana.
#[test]
fn supportive_parents_repeats_only_with_distinct_shared_creatures() {
    let mut s = five_color_unfunded_setup(&[
        "Supportive Parents",
        "Gray Ogre",
        "Badgermole Cub",
        "Scarecrow Guide",
    ]);
    let first = s.force_permanent_on_battlefield(0, "Supportive Parents");
    let second = s.force_permanent_on_battlefield(0, "Supportive Parents");
    let third = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let fourth = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Badgermole Cub");
    let spell = move_to_hand(&mut s, "Scarecrow Guide");
    s.game_mut().scenario_refresh_priority().unwrap();
    command(
        &mut s,
        |action| matches!(action, Action::CastSpell { object, .. } if *object == spell),
    );
    for (source, taps, color) in [(first, [first, third], 1), (first, [second, fourth], 3)] {
        command(
            &mut s,
            |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == source),
        );
        for (index, permanent) in taps.into_iter().enumerate() {
            let observation = Observation::for_player(s.game(), PlayerId(0));
            assert_eq!(
                observation.continuation.last().unwrap().mode,
                Some(2 - index)
            );
            encode(&observation, &ObservationEncoderConfig::default()).unwrap();
            let object = s.game().state.permanents[permanent]
                .as_ref()
                .unwrap()
                .object;
            command(
                &mut s,
                |action| matches!(action, Action::SelectCard { object: selected, .. } if *selected == object),
            );
        }
        assert_eq!(
            s.game().state.players[0].mana_pool.total(),
            if color == 1 { 0 } else { 1 },
            "costs do not produce mana"
        );
        command(
            &mut s,
            |action| matches!(action, Action::ChooseMode { mode, .. } if *mode == color),
        );
    }
    // Both outputs pay the parent spell. Badgermole Cub adds no green mana.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("UR"));
    command(&mut s, |action| matches!(action, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::default());
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Stack));
    assert!([first, second, third, fourth]
        .iter()
        .all(|id| s.game().state.permanents[*id].as_ref().unwrap().tapped));
    assert!(!s.action_space().actions.iter().any(|action| matches!(
        action,
        Action::ActivateAbility { permanent, .. } if *permanent == first || *permanent == second
    )));
}

/// Two sources with one other creature have three creatures, not four taps.
#[test]
fn supportive_parents_cannot_double_count_creatures_across_sources() {
    let mut s = five_color_unfunded_setup(&["Supportive Parents", "Gray Ogre", "Scarecrow Guide"]);
    s.force_permanent_on_battlefield(0, "Supportive Parents");
    s.force_permanent_on_battlefield(0, "Supportive Parents");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let spell = move_to_hand(&mut s, "Scarecrow Guide");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { object, .. } if *object == spell)));
}
