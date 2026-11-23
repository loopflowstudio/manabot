//! Tests for cards filed in `src/chk.rs`, in card order.

use crate::support::inspect::zone_of;
use crate::support::{
    game::{clear_summoning_sickness, five_color_unfunded_setup, move_to_hand, Scenario},
    inspect::{card_of, permanent_named, tapped},
    play::{activate, cast_card, castable, choose, choose_mode},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        ability::Effect,
        card::{ActivatedAbilityDefinition, CardType},
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::Mana,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

fn pool(s: &Scenario, player: usize) -> &Mana {
    &s.game().state.players[player].mana_pool
}

/// Heartbeat of Spring with only `lands` producing mana for player 0.
fn heartbeat_setup(heartbeat_controller: usize, lands: &[&str]) -> (Scenario, Vec<PermanentId>) {
    let mut names = vec![
        "Heartbeat of Spring",
        "Badgermole Cub",
        "Hermitic Herbalist",
    ];
    names.extend(lands.iter().filter(|name| !is_basic(name)));
    let mut s = five_color_unfunded_setup(&names);
    for permanent in s.game_mut().state.permanents.iter_mut().flatten() {
        permanent.tapped = true;
    }
    s.force_permanent_on_battlefield(heartbeat_controller, "Heartbeat of Spring");
    let lands = lands
        .iter()
        .map(|name| s.force_permanent_on_battlefield(0, name))
        .collect();
    (s, lands)
}

fn is_basic(name: &str) -> bool {
    ["Plains", "Island", "Swamp", "Mountain", "Forest"].contains(&name)
}

/// Give `land` an extra "{T}: Add [mana]" (sacrificing it when asked), a
/// shape no printed land has, to exercise several produced types.
fn add_tap_ability(s: &mut Scenario, land: PermanentId, mana: &str, sacrifice: bool) -> usize {
    let card = card_of(s, land);
    let abilities = &mut s.game_mut().state[card].activated_abilities;
    abilities.push(ActivatedAbilityDefinition {
        tap_source: true,
        sacrifice_source: sacrifice,
        effects: vec![Effect::add_mana(Mana::parse(mana))],
        ..Default::default()
    });
    abilities.len() - 1
}

/// A single output is chosen automatically when skipping is on.
fn take_single_output(s: &mut Scenario) {
    if s.action_space().kind == ActionSpaceKind::Modal && s.action_space().actions.len() == 1 {
        choose_mode(s, 0);
    }
}

/// One Forest pays {1}{G}: the Heartbeat's {G} joins the Forest's while the
/// spell's cost is paid, either player's Heartbeat adds to the tapping
/// player's pool, and the bonus is not itself a tap.
#[test]
fn heartbeat_of_spring_adds_one_mana_of_a_produced_type_to_the_tapping_player() {
    for controller in [0, 1] {
        let (mut s, lands) = heartbeat_setup(controller, &["Forest"]);
        assert_eq!(
            s.game_mut().cached_producible_mana(PlayerId(0)),
            Mana::parse("GG"),
            "controller {controller}"
        );
        let cub = move_to_hand(&mut s, "Badgermole Cub");
        assert!(castable(&mut s, cub), "controller {controller}");
        cast_card(&mut s, cub);
        if s.action_space().kind == ActionSpaceKind::PayOrNot {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        assert_eq!(
            zone_of(&s, cub),
            Some(ZoneType::Stack),
            "controller {controller}"
        );
        assert!(tapped(&s, lands[0]));
        assert_eq!(pool(&s, 0).total(), 0);
        assert_eq!(
            pool(&s, 1).total(),
            0,
            "the bonus never goes to its controller"
        );
    }
}

/// The bonus copies the type the ability actually produced: the chosen half
/// of a dual land, and colorless.
#[test]
fn heartbeat_of_spring_copies_the_chosen_or_colorless_type() {
    let (mut s, lands) = heartbeat_setup(1, &["North Pole Gates"]);
    s.allow_priority_floats();
    activate(&mut s, lands[0], 1);
    choose_mode(&mut s, 1);
    assert_eq!(*pool(&s, 0), Mana::parse("UU"));

    let (mut s, _) = heartbeat_setup(0, &["Jasmine Dragon Tea Shop"]);
    s.allow_priority_floats();
    assert_eq!(
        s.game_mut().cached_producible_mana(PlayerId(0)),
        Mana::parse("CC")
    );
}

/// Each Heartbeat is its own triggered mana ability: after a land produces
/// two types, the tapping player chooses each one's type separately, and a
/// choice may differ from the other's.
#[test]
fn two_heartbeats_choose_their_types_separately_after_two_types_are_produced() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    s.allow_priority_floats();
    s.force_permanent_on_battlefield(1, "Heartbeat of Spring");
    let ability = add_tap_ability(&mut s, lands[0], "WU", false);
    activate(&mut s, lands[0], ability);
    take_single_output(&mut s);
    assert_eq!(*pool(&s, 0), Mana::parse("WU"));
    for (pick, expected) in [(0, "WUW"), (1, "WUWU")] {
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().actions.len(), 2, "W or U");
        choose_mode(&mut s, pick);
        assert_eq!(*pool(&s, 0), Mana::parse(expected));
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

/// A land sacrificed to its own mana ability was still tapped for mana: the
/// Heartbeat reads it as it was when tapped.
#[test]
fn heartbeat_of_spring_triggers_for_a_land_sacrificed_in_its_mana_ability() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    s.allow_priority_floats();
    let ability = add_tap_ability(&mut s, lands[0], "G", true);
    activate(&mut s, lands[0], ability);
    take_single_output(&mut s);
    assert!(s.game().state.permanents[lands[0]].is_none());
    assert_eq!(*pool(&s, 0), Mana::parse("GG"));
}

/// Triggered mana abilities are read from the board when the mana is
/// produced (CR 106.12a), so a Heartbeat that the mana ability sacrifices
/// before its output is produced adds nothing.
#[test]
fn heartbeat_of_spring_sacrificed_by_the_mana_ability_adds_nothing() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    s.allow_priority_floats();
    let heartbeat = card_of(&s, permanent_named(&s, "Heartbeat of Spring"));
    let card = card_of(&s, lands[0]);
    let abilities = &mut s.game_mut().state[card].activated_abilities;
    abilities.push(ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![
            Effect::add_mana(Mana::parse("G")),
            Effect::sacrifice(CardPredicate {
                card_type: Some(CardType::Enchantment),
                ..Default::default()
            }),
        ],
        ..Default::default()
    });
    let ability = abilities.len() - 1;
    activate(&mut s, lands[0], ability);
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    take_single_output(&mut s);
    assert_eq!(zone_of(&s, heartbeat), Some(ZoneType::Graveyard));
    assert_eq!(*pool(&s, 0), Mana::parse("G"));
}

/// The bonus type choice names the Heartbeat as its source and the tapped
/// land as context, so it reads apart from the land's own output choice.
#[test]
fn heartbeat_of_spring_type_choice_names_its_source_and_the_tapped_land() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    s.allow_priority_floats();
    let ability = add_tap_ability(&mut s, lands[0], "WU", false);
    activate(&mut s, lands[0], ability);
    take_single_output(&mut s);
    let heartbeat = card_of(&s, permanent_named(&s, "Heartbeat of Spring"));
    let definition = |object: ObjectId| Some(s.game().state[object].definition_id.0);
    let frames = s.game().continuation_frames(PlayerId(0));
    let frame = frames
        .iter()
        .find(|frame| frame.kind == 6 && frame.decision == 3)
        .expect("the type choice's obligation frame");
    assert_eq!(frame.source.definition, definition(heartbeat));
    let land = frame.context.as_ref().expect("the tapped land");
    assert_eq!(land.definition, definition(card_of(&s, lands[0])));
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert!(context.instruction.contains("triggered mana ability"));
}

/// Tapping a nonland for mana triggers nothing; Badgermole Cub's bonus stays
/// with its own controller's creatures.
#[test]
fn heartbeat_of_spring_ignores_nonlands_and_badgermole_stays_with_its_controller() {
    let (mut s, _) = heartbeat_setup(0, &[]);
    s.allow_priority_floats();
    s.force_permanent_on_battlefield(1, "Badgermole Cub");
    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    clear_summoning_sickness(&mut s, herbalist);
    activate(&mut s, herbalist, 0);
    choose_mode(&mut s, 1);
    assert_eq!(*pool(&s, 0), Mana::parse("U"));
}

/// Cast Time Stop from the defending seat over a spell and an activated
/// ability. Cleanup still demands the active player's discard.
#[test]
fn time_stop_exiles_the_stack_during_opposing_combat() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, force_new_card_in_hand},
        inspect::{assert_choice_contract, permanent},
        play::{attack_with, cast, resolve_top, target_opponent},
    };
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&[
        "Time Stop",
        "Sundial of the Infinite",
        "Lightning Bolt",
        "Gray Ogre",
    ]);
    let sundial = s.force_permanent_on_battlefield(0, "Sundial of the Infinite");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, ogre);
    assert!(permanent(&s, ogre).attacking);
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(managym::flow::turn::AdditionalCombat { attackers: None });
    s.game_mut().state.turn.extra_turns.push(PlayerId(0));
    s.set_player_mana_pool(0, "1R");
    s.set_player_mana_pool(1, "4UU");
    let bolt = cast(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    activate(&mut s, sundial, 0);
    s.pass_priority();
    let stop = force_new_card_in_hand(&mut s, 1, "Time Stop");
    cast_card(&mut s, stop);
    assert_eq!(s.game().state.stack_objects.len(), 3);
    for land in ["Forest", "Island"].into_iter().cycle().take(8) {
        force_new_card_in_hand(&mut s, 0, land);
    }
    let turn = s.game().state.turn.turn_number;
    resolve_top(&mut s);
    assert_eq!(zone_of(&s, stop), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Exile));
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.combat.is_none());
    assert!(!permanent(&s, ogre).attacking);
    assert!(s.game().state.turn.additional_combats.is_empty());
    assert_eq!(s.current_step(), StepKind::Cleanup);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert_choice_contract(&s);
    s.step_action(0);
    assert_eq!(s.game().state.turn.turn_number, turn + 1);
    assert_eq!(s.active_player(), 0, "the queued extra turn is preserved");
    s.assert_life(1, 20);
}

/// Unearth's delayed trigger waits for an end step that actually begins.
#[test]
fn time_stop_skips_the_end_step_but_preserves_its_delayed_trigger() {
    use crate::support::{
        game::{five_color_setup, plant_in_graveyard},
        play::{activate_card, cast, resolve_top, settle, spend_colorless},
    };
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&["Time Stop", "Cityscape Leveler"]);
    let leveler = plant_in_graveyard(&mut s, 0, "Cityscape Leveler");
    activate_card(&mut s, leveler, 0);
    spend_colorless(&mut s);
    settle(&mut s);
    cast(&mut s, "Time Stop");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.active_player(), 1);
    assert_eq!(zone_of(&s, leveler), Some(ZoneType::Battlefield));
    s.advance_to_active_step(1, StepKind::End);
    resolve_top(&mut s);
    assert_eq!(zone_of(&s, leveler), Some(ZoneType::Exile));
}
