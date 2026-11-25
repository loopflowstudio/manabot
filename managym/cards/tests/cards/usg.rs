//! Urza's Saga card behavior.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, set_counters},
    play::{attack_with, command, pass_until},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{game_object::PlayerId, permanent::CounterKind},
};

#[test]
fn fog_bank_blocks_a_lifelinker_in_both_replacement_orders() {
    let mut s = five_color_setup(&["Fog Bank", "Healer's Hawk", "Twinflame Tyrant"]);
    let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    s.force_permanent_on_battlefield(0, "Twinflame Tyrant");
    let fog = s.force_permanent_on_battlefield(1, "Fog Bank");
    set_counters(&mut s, fog, CounterKind::PlusOnePlusOne, 1);
    clear_summoning_sickness(&mut s, hawk);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, hawk);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.action_space().actions.len(), 2);
    for mode in 0..2 {
        let mut branch = s.clone();
        let (mut replay, _) = command(
            &mut branch,
            |action| matches!(action, Action::ChooseMode {mode: choice, ..} if *choice == mode),
        );
        for game in [&mut branch, &mut replay] {
            game.advance_to_active_step(0, StepKind::EndOfCombat);
            assert_eq!(game.life(0), 20);
            for creature in [hawk, fog] {
                assert_eq!(
                    game.game().state.permanents[creature]
                        .as_ref()
                        .unwrap()
                        .damage,
                    0
                );
            }
        }
    }
}

#[test]
fn foundations_discard_duress() {
    crate::assert_revealed_discard("Duress", &["Lightning Bolt", "Giant Growth"]);
}

#[test]
fn confiscate_competes_with_later_control_and_releases_its_permanent() {
    use crate::support::{
        game::move_card,
        inspect::{card_of, permanent, permanent_named},
        play::{cast, settle, spend_colorless},
    };
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
            observation_layout::permanent as columns,
        },
        state::{
            effect::{EffectDuration, Modification},
            game_object::Target,
            zone::ZoneType,
        },
    };
    let mut s = five_color_setup(&["Confiscate", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    s.force_permanent_on_battlefield(1, "Island");
    cast(&mut s, "Confiscate");
    let (mut fork, _) = command(
        &mut s,
        |a| matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == ogre),
    );
    for s in [&mut s, &mut fork] {
        spend_colorless(s);
        settle(s);
        assert_eq!(permanent(s, ogre).controller, PlayerId(0));
        assert!(permanent(s, ogre).summoning_sick);
        let view = Observation::for_player(s.game(), PlayerId(1));
        let row = view
            .opponent_permanents
            .iter()
            .position(|p| p.id == permanent(s, ogre).id.0 as i32)
            .unwrap();
        assert!(view.opponent_permanents[row].control_differs_from_entry);
        let encoded = encode(&view, &ObservationEncoderConfig::default()).unwrap();
        assert_eq!(
            encoded.opponent_permanents[row * columns::WIDTH + columns::CONTROL_DIFFERS_FROM_ENTRY],
            1.0
        );
        let affected = s.game().affected_permanent(ogre);
        s.game_mut().apply_continuous(
            None,
            PlayerId(1),
            affected,
            Modification::Control,
            EffectDuration::EndOfTurn,
        );
        assert_eq!(permanent(s, ogre).controller, PlayerId(1));
        s.advance_to_active_step(1, StepKind::Upkeep);
        assert_eq!(permanent(s, ogre).controller, PlayerId(0));
        let aura = card_of(s, permanent_named(s, "Confiscate"));
        move_card(s, aura, ZoneType::Graveyard);
        assert_eq!(permanent(s, ogre).controller, PlayerId(1));
        assert!(permanent(s, ogre).summoning_sick);
    }
    assert_eq!(
        s.game().state.deterministic_hash(),
        fork.game().state.deterministic_hash()
    );
}

#[test]
fn confiscate_follows_its_own_controller_even_when_it_is_the_older_aura() {
    use crate::support::{
        game::move_card,
        inspect::{card_of, permanent, permanent_named},
        play::{cast, settle},
    };
    use managym::state::{
        effect::{EffectDuration, Modification},
        game_object::Target,
        zone::ZoneType,
    };
    let mut s = five_color_setup(&["Confiscate"]);
    let land = s.force_permanent_on_battlefield(1, "Island");
    s.force_permanent_on_battlefield(0, "Forest");
    s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
    cast(&mut s, "Confiscate");
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    let aura = permanent_named(&s, "Confiscate");
    assert_eq!(permanent(&s, land).controller, PlayerId(0));
    let affected = s.game().affected_permanent(aura);
    s.game_mut().apply_continuous(
        None,
        PlayerId(1),
        affected,
        Modification::Control,
        EffectDuration::Indefinite,
    );
    assert_eq!(permanent(&s, land).controller, PlayerId(1));
    let card = card_of(&s, aura);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(permanent(&s, land).controller, PlayerId(1));
    assert!(permanent(&s, land).tapped);
}

#[test]
fn confiscates_departure_removes_the_blocker_but_the_attacker_stays_blocked() {
    use crate::support::{
        game::{move_card, move_to_hand},
        inspect::{card_of, permanent, permanent_named},
        play::{cast, settle},
    };
    use managym::{
        agent::observation::Observation,
        state::{game_object::Target, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Confiscate", "Gray Ogre", "Hill Giant", "Giant Growth"]);
    let blocker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let attacker = s.force_permanent_on_battlefield(1, "Hill Giant");
    cast(&mut s, "Confiscate");
    assert!(s.choose_target(Target::Permanent(blocker)));
    settle(&mut s);
    s.force_permanent_on_battlefield(0, "Forest");
    move_to_hand(&mut s, "Giant Growth");
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    attack_with(&mut s, attacker);
    s.advance_to_active_step(1, StepKind::DeclareBlockers);
    s.declare_block();
    let aura = card_of(&s, permanent_named(&s, "Confiscate"));
    move_card(&mut s, aura, ZoneType::Graveyard);
    assert_eq!(permanent(&s, blocker).controller, PlayerId(1));
    let obs = Observation::for_player(s.game(), PlayerId(1));
    let row = |id| {
        obs.agent_permanents
            .iter()
            .find(|p| p.id == permanent(&s, id).id.0 as i32)
            .unwrap()
    };
    assert!(row(attacker).blocked);
    assert_eq!(row(blocker).blocking, None);
    s.advance_to_active_step(1, StepKind::EndOfCombat);
    assert_eq!(s.life(0), 20);
    assert_eq!(permanent(&s, attacker).damage, 0);
    assert_eq!(permanent(&s, blocker).damage, 0);
}

#[test]
fn blanchwood_armor_counts_the_auras_controllers_forests() {
    use crate::support::play::{cast, settle};
    use managym::state::{
        effect::{EffectDuration, Modification},
        game_object::Target,
    };
    let mut s = five_color_setup(&["Blanchwood Armor", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Blanchwood Armor");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (3, 3));
    let affected = s.game().affected_permanent(ogre);
    s.game_mut().apply_continuous(
        None,
        PlayerId(1),
        affected,
        Modification::Control,
        EffectDuration::Indefinite,
    );
    assert_eq!(s.game().characteristics(ogre).pt(), (3, 3));
    s.force_permanent_on_battlefield(0, "Forest");
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
}
