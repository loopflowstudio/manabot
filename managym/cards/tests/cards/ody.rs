//! Tests for cards filed in `src/ody.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, five_color_unfunded_setup, move_to_hand},
    inspect::{card_of, has_action, library_card, now, offered_cards, top_spell, zone_of},
    play::{
        activate, cast, cast_card, cast_owned, castable, choose, choose_mode, select_card, settle,
        spend_named,
    },
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        ability::Effect,
        card::ActivatedAbilityDefinition,
        effect::{Affected, EffectDuration, Modification},
        game_object::{PermanentId, PlayerId},
        mana::{Color, Mana, ManaCost},
        zone::ZoneType,
    },
};

#[test]
fn entomb_puts_any_chosen_library_card_into_the_graveyard() {
    let mut s = five_color_setup(&["Entomb", "Gray Ogre"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    cast_owned(&mut s, "Entomb");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    assert!(offered_cards(&s).contains(&ogre));
    let library = s.zone_size(0, ZoneType::Library);
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
    s.assert_zone_size(0, ZoneType::Library, library - 1);
}

/// The caster's opponent draws: the controller when the opponent casts, the
/// opponent when the controller casts.
#[test]
fn standstill_sacrifices_and_the_casters_opponent_draws_three() {
    for caster in [0usize, 1] {
        let mut s = five_color_setup(&["Standstill", "Lightning Bolt"]);
        let standstill = s.force_permanent_on_battlefield(0, "Standstill");
        if caster == 1 {
            s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
            s.set_player_mana_pool(1, "R");
        }
        let bolt = crate::support::game::force_new_card_in_hand(&mut s, caster, "Lightning Bolt");
        s.game_mut().scenario_refresh_priority().unwrap();
        let hands = [
            s.zone_size(0, ZoneType::Hand),
            s.zone_size(1, ZoneType::Hand),
        ];
        crate::support::play::cast_card(&mut s, bolt);
        crate::support::play::target_opponent(&mut s);
        settle(&mut s);
        assert!(s.game().state.permanents[standstill].is_none());
        let opponent = 1 - caster;
        s.assert_zone_size(opponent, ZoneType::Hand, hands[opponent] + 3);
        s.assert_zone_size(caster, ZoneType::Hand, hands[caster] - 1);
    }
}

/// Its {C} ability is an ordinary fixed mana ability that deals no damage;
/// the any-color ability deals 3 damage to its controller as part of the same
/// mana ability, once, from the land itself, so protection prevents it.
#[test]
fn tarnished_citadel_deals_three_damage_only_with_colored_mana() {
    let mut s = five_color_unfunded_setup(&["Tarnished Citadel", "Sol Ring"]);
    s.allow_priority_floats();
    s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    let ring = move_to_hand(&mut s, "Sol Ring");
    cast_card(&mut s, ring);
    assert!(
        has_action(&s, |a| matches!(a, Action::ActivateAbility { .. })),
        "the colored ability is offered only as an explicit choice"
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Sol Ring").len(), 1);
    assert_eq!(s.life(0), 20, "{{C}} paid Sol Ring without damage");

    for protected in [false, true] {
        let mut s = five_color_unfunded_setup(&["Tarnished Citadel"]);
        s.allow_priority_floats();
        let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
        if protected {
            s.game_mut().apply_continuous(
                None,
                PlayerId(0),
                [Affected::Player(PlayerId(0))],
                Modification::ProtectionFromEverything,
                EffectDuration::UntilYourNextTurn,
            );
        }
        activate(&mut s, citadel, 0);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().actions.len(), 5);
        choose_mode(&mut s, 3);
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("R"));
        assert!(s.game().state.stack_objects.is_empty(), "no stack");
        assert_eq!(s.life(0), if protected { 20 } else { 17 });
        settle(&mut s);
        assert_eq!(s.life(0), if protected { 20 } else { 17 }, "damage once");
    }
}

/// Under Teferi's Protection the colored activation still adds its mana and
/// its damage is prevented.
#[test]
fn tarnished_citadel_damage_is_prevented_under_teferis_protection() {
    let mut s =
        five_color_unfunded_setup(&["Tarnished Citadel", "Teferi's Protection", "Sol Ring"]);
    s.allow_priority_floats();
    s.set_player_mana_pool(0, "2W");
    cast(&mut s, "Teferi's Protection");
    settle(&mut s);
    let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    assert!(s.game().player_has(
        PlayerId(0),
        &managym::state::effect::Modification::ProtectionFromEverything
    ));
    activate(&mut s, citadel, 0);
    choose_mode(&mut s, 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("W"));
    assert_eq!(s.life(0), 20);
}

fn mana_modes(s: &crate::support::game::Scenario) -> Vec<usize> {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    s.action_space()
        .actions
        .iter()
        .map(|action| match action {
            Action::ChooseMode { mode, .. } => *mode,
            action => panic!("expected a mana color, got {action:?}"),
        })
        .collect()
}

#[test]
fn tarnished_citadel_and_heartbeat_complete_manual_payment() {
    let spell = "Sokka, Bold Boomeranger";
    for (second, life, citadel_first, citadel_color, second_color) in [
        ("Airship Engine Room", 2, true, 1, 1),
        ("Airship Engine Room", 2, true, 3, 0),
        ("Airship Engine Room", 3, true, 1, 1),
        ("Sunbaked Canyon", 3, false, 1, 0),
        ("Sunbaked Canyon", 5, true, 1, 0),
        ("Sunbaked Canyon", 5, false, 1, 0),
    ] {
        let room = second == "Airship Engine Room";
        let mut s =
            five_color_unfunded_setup(&[spell, "Tarnished Citadel", second, "Heartbeat of Spring"]);
        s.game_mut().skip_trivial = false;
        let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
        let other = s.force_permanent_on_battlefield(0, second);
        let other_ability = usize::from(room);
        s.force_permanent_on_battlefield(0, "Heartbeat of Spring");
        s.game_mut().scenario_set_life(PlayerId(0), life);

        cast_owned(&mut s, spell);
        let offered = |permanent, index| {
            has_action(&s, |action| {
                matches!(action, Action::ActivateAbility { permanent: p, ability_index, .. }
                if *p == permanent && *ability_index == index)
            })
        };
        assert_eq!(offered(citadel, 0), room || life > 3);
        assert!(offered(other, other_ability));
        if citadel_first {
            activate(&mut s, citadel, 0);
            assert_eq!(s.life(0), life - 3);
            assert_eq!(mana_modes(&s), if room { vec![1, 3] } else { vec![1] });
            choose_mode(&mut s, citadel_color);
            activate(&mut s, other, other_ability);
            assert_eq!(mana_modes(&s), [second_color]);
            choose_mode(&mut s, second_color);
        } else {
            activate(&mut s, other, other_ability);
            assert_eq!(s.life(0), life - 1);
            assert_eq!(mana_modes(&s), [second_color]);
            choose_mode(&mut s, second_color);
            activate(&mut s, citadel, 0);
            assert_eq!(mana_modes(&s), [citadel_color]);
            choose_mode(&mut s, citadel_color);
        }
        let ending_life = life - 3 - i32::from(!room);
        assert_eq!(s.life(0), ending_life);
        for land in [citadel, other] {
            assert!(s.game().state.permanents[land].as_ref().unwrap().tapped);
        }
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("UR"));
        assert_eq!(s.game().state.stack_objects.len(), 1);
        assert_eq!(
            s.game().state.loss(0).is_some_and(|loss| loss.life_total),
            ending_life <= 0
        );
    }
}

#[test]
fn tarnished_citadel_can_gain_life_before_paying_canyon() {
    // A colorless Citadel tap cannot pay one of these three colored pips.
    for protected in [false, true] {
        let spell = "Sokka, Bold Boomeranger";
        let mut s = five_color_unfunded_setup(&[
            spell,
            "Tarnished Citadel",
            "The Great Henge",
            "Sunbaked Canyon",
        ]);
        s.game_mut().skip_trivial = false;
        let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
        let henge = (!protected).then(|| s.force_permanent_on_battlefield(0, "The Great Henge"));
        let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
        s.game_mut().scenario_set_life(PlayerId(0), 2);
        if protected {
            s.game_mut().apply_continuous(
                None,
                PlayerId(0),
                [Affected::Player(PlayerId(0))],
                Modification::ProtectionFromEverything,
                EffectDuration::UntilYourNextTurn,
            );
        }
        let object = move_to_hand(&mut s, spell);
        if !protected {
            s.game_mut().state[object].mana_cost = Some(ManaCost::parse("UGGR"));
        }
        let mut fork = s.clone();
        let before = s.game().state.deterministic_hash();
        assert!(castable(&mut s, object));
        assert_eq!(s.game().state.deterministic_hash(), before);
        cast_card(&mut fork, object);
        assert_eq!(s.game().state.deterministic_hash(), before);
        assert_eq!(s.life(0), 2);
        assert!(!s.game().state.permanents[citadel].as_ref().unwrap().tapped);

        cast_card(&mut s, object);
        activate(&mut s, citadel, 0);
        assert_eq!(s.life(0), if protected { 2 } else { -1 });
        assert_eq!(mana_modes(&s), [1]);
        choose_mode(&mut s, 1);
        if let Some(henge) = henge {
            activate(&mut s, henge, 0);
            assert_eq!(mana_modes(&s), [0]);
            choose_mode(&mut s, 0);
            assert_eq!(s.life(0), 1);
        }
        activate(&mut s, canyon, 0);
        choose_mode(&mut s, 0);
        assert_eq!(s.life(0), if protected { 1 } else { 0 });
        assert_eq!(s.game().state.stack_objects.len(), 1);
    }
}

#[test]
fn tarnished_citadel_still_costs_three_life_beside_angel_of_vitality() {
    let spell = "Sokka, Bold Boomeranger";
    let mut s = five_color_unfunded_setup(&[
        spell,
        "Tarnished Citadel",
        "Sunbaked Canyon",
        "Angel of Vitality",
    ]);
    s.game_mut().skip_trivial = false;
    let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    s.force_permanent_on_battlefield(0, "Angel of Vitality");
    s.game_mut().scenario_set_life(PlayerId(0), 3);

    cast_owned(&mut s, spell);
    assert_eq!(
        s.action_space().actions,
        [Action::ActivateAbility {
            player: PlayerId(0),
            permanent: canyon,
            ability_index: 0,
        }]
    );
    activate(&mut s, canyon, 0);
    choose_mode(&mut s, 0); // Red.
    activate(&mut s, citadel, 0);
    choose_mode(&mut s, 1); // Blue.
    assert_eq!(s.life(0), -1);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert!(s.game().state.loss(0).is_some_and(|loss| loss.life_total));
}

fn heartbeat_modal_land() -> (crate::support::game::Scenario, PermanentId) {
    let mut s = five_color_unfunded_setup(&[
        "Sokka, Bold Boomeranger",
        "Tarnished Citadel",
        "Heartbeat of Spring",
    ]);
    s.game_mut().skip_trivial = false;
    let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    s.force_permanent_on_battlefield(0, "Heartbeat of Spring");
    let source = card_of(&s, citadel);
    s.game_mut().state[source].activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![Effect::mana_choice(["UR", "UG"].map(Mana::parse))],
        ..Default::default()
    }];
    (s, citadel)
}

#[test]
fn heartbeat_bonuses_commit_the_original_output_with_its_source() {
    let (mut s, citadel) = heartbeat_modal_land();
    s.force_permanent_on_battlefield(0, "Heartbeat of Spring");
    s.set_player_mana_pool(0, "U");
    let source = card_of(&s, citadel);
    let spell = move_to_hand(&mut s, "Sokka, Bold Boomeranger");

    cast_card(&mut s, spell);
    activate(&mut s, citadel, 0);
    choose_mode(&mut s, 0); // Produce {U}{R}.
    choose_mode(&mut s, 1); // First Heartbeat adds {R}.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("UURR"));
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(mana_modes(&s), [0, 1]);
    choose_mode(&mut s, 0); // Second Heartbeat adds {U}.

    let cast = top_spell(&s);
    assert_eq!(cast.object, now(&s, spell));
    let spent = cast.spent.mana();
    assert_eq!(
        spent.iter().map(|unit| unit.color).collect::<Vec<_>>(),
        [Color::Blue as u8, Color::Red as u8]
    );
    assert!(spent
        .iter()
        .all(|unit| unit.source.is_some_and(|origin| origin.object == source)));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("UUR"));
}

#[test]
fn heartbeat_bonus_keeps_the_generic_spend_choice() {
    let (mut s, citadel) = heartbeat_modal_land();
    let object = move_to_hand(&mut s, "Sokka, Bold Boomeranger");
    s.game_mut().state[object].mana_cost = Some(ManaCost::parse("1U"));

    cast_card(&mut s, object);
    activate(&mut s, citadel, 0);
    choose_mode(&mut s, 0); // Produce {U}{R}.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("UR"));
    assert_eq!(mana_modes(&s), [0, 1]);
    choose_mode(&mut s, 0); // Heartbeat adds {U}.
    spend_named(&mut s, |label| label.contains("{U}"));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("R"));
}

#[test]
fn the_great_henge_funds_an_announced_life_fee_after_canyon() {
    let mut s = five_color_unfunded_setup(&["Gray Ogre", "Sunbaked Canyon", "The Great Henge"]);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    let henge = s.force_permanent_on_battlefield(0, "The Great Henge");
    let object = s.game().state.permanents[source].as_ref().unwrap().object;
    s.game_mut().state[object].activated_abilities = vec![ActivatedAbilityDefinition {
        mana_cost: ManaCost::parse("GGR"),
        life_cost: 2,
        effects: vec![Effect::gain_life(1)],
        ..Default::default()
    }];
    s.game_mut().skip_trivial = false;
    s.game_mut().scenario_set_life(PlayerId(0), 2);

    activate(&mut s, source, 0);
    activate(&mut s, canyon, 0);
    assert_eq!(s.life(0), 1);
    assert_eq!(mana_modes(&s), [0]);
    choose_mode(&mut s, 0);
    activate(&mut s, henge, 0);
    assert_eq!(s.life(0), 3);
    choose_mode(&mut s, 0);
    assert_eq!(s.life(0), 1);
    assert_eq!(s.game().state.stack_objects.len(), 1);
}

#[test]
fn tarnished_citadel_and_the_great_henge_change_another_mana_source() {
    let player = PlayerId(0);
    for (name, life, count) in [
        ("Tarnished Citadel", 28, 2),
        ("The Great Henge", 22, 1),
        ("Sunbaked Canyon", 25, 1),
    ] {
        let mut s = five_color_unfunded_setup(&[
            name,
            "Island",
            "Angel of Vitality",
            "Raucous Audience",
            "Solemn Simulacrum",
        ]);
        let sources: Vec<_> = (0..count)
            .map(|_| s.force_permanent_on_battlefield(0, name))
            .collect();
        let audience = s.force_permanent_on_battlefield(0, "Raucous Audience");
        clear_summoning_sickness(&mut s, audience);
        s.force_permanent_on_battlefield(0, "Angel of Vitality");
        let island =
            (name == "Sunbaked Canyon").then(|| s.force_permanent_on_battlefield(0, "Island"));
        s.game_mut().scenario_set_life(player, life);
        s.allow_priority_floats();
        let spell = move_to_hand(&mut s, "Solemn Simulacrum");
        assert!(castable(&mut s, spell), "{name} at {life}");
        cast_card(&mut s, spell);
        let offered = |s: &_, source| {
            has_action(s, |action| {
                matches!(action, Action::ActivateAbility { permanent, ability_index: 0, .. }
                if *permanent == source)
            })
        };
        assert_eq!(offered(&s, audience), name != "The Great Henge");
        assert_eq!(offered(&s, sources[0]), name != "Sunbaked Canyon");
        match name {
            "Tarnished Citadel" => {
                activate(&mut s, sources[0], 0);
                choose_mode(&mut s, 1);
                assert!(!offered(&s, sources[1]));
                activate(&mut s, audience, 0);
                choose_mode(&mut s, 0);
                activate(&mut s, sources[1], 0);
                choose_mode(&mut s, 0);
                assert_eq!(s.life(0), 22);
            }
            "The Great Henge" => {
                activate(&mut s, sources[0], 0);
                choose_mode(&mut s, 0);
                activate(&mut s, audience, 0);
                choose_mode(&mut s, 0);
                assert_eq!(s.life(0), 25);
            }
            "Sunbaked Canyon" => {
                activate(&mut s, audience, 0);
                choose_mode(&mut s, 0);
                activate(&mut s, sources[0], 0);
                choose_mode(&mut s, 0);
                let permanent = island.unwrap();
                choose(&mut s, |action| {
                    *action == Action::TapForMana { player, permanent }
                });
                assert_eq!(s.life(0), 24);
            }
            _ => unreachable!(),
        }
        assert_eq!(top_spell(&s).object, now(&s, spell));
    }
}

#[test]
fn foundations_5e_zombify() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, offered_targets, permanent, permanent_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Zombify", "Gray Ogre"]);
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let ours = card_of(&s, ours);
    let ours = move_card(&mut s, ours, ZoneType::Graveyard);
    let theirs = card_of(&s, theirs);
    let theirs = move_card(&mut s, theirs, ZoneType::Graveyard);
    let land = card_of(&s, land);
    let land = move_card(&mut s, land, ZoneType::Graveyard);
    cast(&mut s, "Zombify");
    let offers = offered_targets(&s);
    assert_eq!(offers, [Target::GraveyardCard(ours)]);
    assert!(!offers.contains(&Target::GraveyardCard(theirs)));
    assert!(!offers.contains(&Target::GraveyardCard(land)));
    assert!(s.choose_target(Target::GraveyardCard(ours)));
    settle(&mut s);
    let returned = permanent_of(&s, ours).expect("returned");
    assert!(!permanent(&s, returned).tapped);
}
