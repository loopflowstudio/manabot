//! Tests for cards filed in `src/tdm.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand, plant_in_graveyard},
    inspect::{can_cast, card_of, now, offered_cards, top_spell},
    play::{cast_card, select_card, settle},
};
use managym::state::{
    game_object::PlayerId,
    mana::Color,
    payment::{How, Toward},
    player::ManaPayment,
    zone::ZoneType,
};

/// Teval gives every spell its controller casts delve. Each graveyard card
/// exiled pays {1} and never a colored pip; the receipt names the exiled
/// cards; Teval's life loss is the spell's mana value, not the mana paid.
#[test]
fn teval_grants_delve_and_takes_life_equal_to_mana_value() {
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Teval, Arbiter of Virtue",
            "Gray Ogre",
            "Hill Giant",
            "Savannah Lions",
            "Llanowar Elves",
        ]);
        s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
        let mountain = s.force_permanent_on_battlefield(0, "Mountain");
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
        // {2}{R} with one Mountain: not castable until the graveyard can
        // pay the generic part.
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&s, ogre), "auto={auto}");
        let graveyard: Vec<_> = ["Hill Giant", "Savannah Lions", "Llanowar Elves"]
            .into_iter()
            .map(|name| {
                let card = plant_in_graveyard(&mut s, 0, name);
                now(&s, card)
            })
            .collect();
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(can_cast(&s, ogre), "auto={auto}");

        cast_card(&mut s, ogre);
        // Every graveyard card is offered, and paying is not: the
        // Mountain alone cannot cover the cost. Which cards to exile is
        // the player's pick under the automatic payer too.
        let mut offered = offered_cards(&s);
        offered.sort();
        let mut expected = graveyard.clone();
        expected.sort();
        assert_eq!(offered, expected);
        let frame = s.game().semantic_decision_frame().unwrap();
        assert!(
            frame
                .offers
                .iter()
                .any(|offer| offer.label.ends_with("to pay {1} (delve)")),
            "{:?}",
            frame.offers
        );
        select_card(&mut s, graveyard[0]);
        // A fork at the second delve choice: the card not chosen stays.
        for (chosen, kept) in [(1, 2), (2, 1)] {
            let fork = crate::support::play::fork_selecting(&s, graveyard[chosen]);
            assert_eq!(
                top_spell(&fork)
                    .spent
                    .contributed(How::Exiled, Toward::Delve)
                    .count(),
                2
            );
            let in_graveyard = |object| {
                fork.game()
                    .state
                    .zones
                    .contains(object, ZoneType::Graveyard, PlayerId(0))
            };
            assert!(in_graveyard(graveyard[kept]) && !in_graveyard(graveyard[chosen]));
        }
        select_card(&mut s, graveyard[2]);
        // With {R} left and one Mountain, the payment finishes itself.
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 1, "auto={auto}");
        assert_eq!(spell.spent.amounts()[Color::Red as usize], 1);
        assert_eq!(
            spell.spent.mana()[0].source.unwrap().object,
            card_of(&s, mountain)
        );
        let exiled: Vec<_> = spell
            .spent
            .contributed(How::Exiled, Toward::Delve)
            .collect();
        assert_eq!(exiled.len(), 2, "auto={auto}");
        assert_eq!(spell.spent.contributions().len(), 2);
        for object in &exiled {
            assert!(s
                .game()
                .state
                .zones
                .contains(*object, ZoneType::Exile, PlayerId(0)));
        }
        s.assert_zone_size(0, ZoneType::Graveyard, 1);
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.starts_with("Paid with: exiled ") && line.ends_with(" to delve.")),
            "{:?}",
            context.pending
        );
        settle(&mut s);
        // Mana value three, although one mana was paid.
        s.assert_life(0, 17);

        if auto {
            // With lands for the whole cost the automatic payer pays with
            // mana: nothing is asked and the last graveyard card stays.
            for _ in 0..3 {
                s.force_permanent_on_battlefield(0, "Mountain");
            }
            let second = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
            s.game_mut().scenario_refresh_priority().unwrap();
            cast_card(&mut s, second);
            let spell = top_spell(&s);
            assert_eq!(spell.spent.total(), 3);
            assert!(spell.spent.contributions().is_empty());
            s.assert_zone_size(0, ZoneType::Graveyard, 1);
        }
    }
}

/// Delve pays generic mana only: no number of graveyard cards casts a spell
/// whose colored mana the caster cannot produce.
#[test]
fn teval_delve_never_pays_a_colored_pip() {
    let mut s = five_color_unfunded_setup(&["Teval, Arbiter of Virtue", "Gray Ogre", "Hill Giant"]);
    s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
    for _ in 0..4 {
        plant_in_graveyard(&mut s, 0, "Hill Giant");
    }
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre));
}

/// A spell cast from the graveyard is not a card its own delve can exile
/// (CR 601.2a: it is already on the stack as costs are paid).
#[test]
fn teval_delve_never_exiles_the_spell_being_cast_from_the_graveyard() {
    let mut s = five_color_unfunded_setup(&[
        "Teval, Arbiter of Virtue",
        "Join the Dance",
        "Hill Giant",
        "Gray Ogre",
    ]);
    s.force_permanent_on_battlefield(0, "Teval, Arbiter of Virtue");
    s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Plains");
    let dance = plant_in_graveyard(&mut s, 0, "Join the Dance");
    plant_in_graveyard(&mut s, 0, "Hill Giant");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    // Flashback {3}{G}{W} with two lands and two other graveyard cards: one
    // generic short.
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, dance));

    let third = plant_in_graveyard(&mut s, 0, "Hill Giant");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, dance));
    cast_card(&mut s, dance);
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), 3, "{offered:?}");
    assert!(!offered.contains(&now(&s, dance)));
    assert!(offered.contains(&now(&s, third)));
}

#[test]
fn vizier_filters_twinmaw_profiles_and_mana() {
    use crate::support::{
        game::Scenario,
        game::{move_card, move_from_library},
        play::choose,
    };
    use managym::{agent::action::Action, state::card::SpellProfile};
    fn profiles(s: &Scenario, card: managym::state::game_object::ObjectId) -> Vec<SpellProfile> {
        s.action_space()
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::CastSpell {
                    object, profile, ..
                } if *object == card => Some(*profile),
                _ => None,
            })
            .collect()
    }

    let creature = "Twinmaw Stormbrood";
    let mut s = five_color_unfunded_setup(&[creature, "Vizier of the Menagerie", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Vizier of the Menagerie");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let id = move_from_library(&mut s, 0, creature, ZoneType::Exile);
    let id = move_card(&mut s, id, ZoneType::Library);
    s.set_player_mana_pool(0, "RRRRRR");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert_eq!(profiles(&s, id), [SpellProfile::Front]);
    let id = move_card(&mut s, id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert_eq!(profiles(&s, id), [SpellProfile::Front, SpellProfile::Omen]);
    s.set_player_mana_pool(0, "UUUUUU");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert_eq!(profiles(&s, id), [SpellProfile::Front]);
    let life = s.game().state.players[0].life;
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell {object, profile: SpellProfile::Front,..} if *object == id),
    );
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, life + 5);
}

/// Give only the Omen an exile alternative. Its front still requires six mana.
#[test]
fn charring_bite_uses_the_selected_profiles_alternative_cost() {
    use crate::support::{
        inspect::{offer_labels, zone_of},
        play::choose,
    };
    use managym::{
        agent::action::Action,
        state::{
            card::{AdditionalCost, InsetSpell, SpellProfile},
            game_object::Target,
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_unfunded_setup(&["Twinmaw Stormbrood", "Gray Ogre", "Lightning Bolt"]);
    let dragon = force_new_card_in_hand(&mut s, 0, "Twinmaw Stormbrood");
    let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
    force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let target = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let Some(InsetSpell::Omen(omen)) = &mut s.game_mut().state[dragon].inset else {
        unreachable!()
    };
    std::sync::Arc::make_mut(omen).additional_costs.push(
        AdditionalCost::ExileFromHandAlternative {
            predicate: CardPredicate {
                colors_any: vec![Color::Red],
                ..Default::default()
            },
            not_your_turn: false,
        },
    );
    s.set_player_mana_pool(0, "RR");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { object, profile: SpellProfile::Omen, .. } if *object == dragon),
    );
    assert!(offer_labels(&s)
        .iter()
        .any(|label| label.starts_with("Exile a card")));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert!(s.choose_target(Target::Permanent(target)));
    select_card(&mut s, bolt);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Exile));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
    settle(&mut s);
    assert_eq!(zone_of(&s, dragon), Some(ZoneType::Library));
}

#[test]
fn charring_bite_shuffles_only_on_resolution() {
    use crate::support::{
        game::{force_lands, move_from_library},
        inspect::zone_of,
        play::{choose, fork_choosing},
    };
    use managym::{
        agent::action::Action,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s = five_color_unfunded_setup(&[
        "Twinmaw Stormbrood",
        "Gray Ogre",
        "Lightning Bolt",
        "Counterspell",
    ]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let flyer = s.force_permanent_on_battlefield(1, "Twinmaw Stormbrood");
    let target = card_of(&s, ogre);
    let spell = move_from_library(&mut s, 0, "Twinmaw Stormbrood", ZoneType::Hand);
    let bolt = move_from_library(&mut s, 1, "Lightning Bolt", ZoneType::Hand);
    let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
    force_lands(&mut s, 0, "Mountain", 2);
    force_lands(&mut s, 1, "Island", 2);
    force_lands(&mut s, 1, "Mountain", 1);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a,Action::CastSpell {object,profile:SpellProfile::Omen,..} if *object==spell),
    );
    assert!(!s.choose_target(Target::Permanent(flyer)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    let rng = s.game().state.rng.get_word_pos();
    let mut lost = fork_choosing(
        &s,
        |a| matches!(a,Action::CastSpell{object,..} if *object==bolt),
    );
    assert!(lost.choose_target(Target::Permanent(ogre)));
    settle(&mut lost);
    assert_eq!(zone_of(&lost, spell), Some(ZoneType::Graveyard));
    assert_eq!(lost.game().state.rng.get_word_pos(), rng);
    let mut countered = fork_choosing(
        &s,
        |a| matches!(a,Action::CastSpell{object,..} if *object==counter),
    );
    let stack_id = now(&countered, spell);
    assert!(countered.choose_target(Target::StackSpell(stack_id)));
    settle(&mut countered);
    assert_eq!(zone_of(&countered, spell), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&countered, target), Some(ZoneType::Battlefield));
    assert_eq!(countered.game().state.rng.get_word_pos(), rng);
    let mut replay = fork_choosing(&s, |a| matches!(a, Action::PassPriority { .. }));
    settle(&mut replay);
    settle(&mut s);
    assert_eq!(
        s.game().state.rng.get_word_pos(),
        replay.game().state.rng.get_word_pos()
    );
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0)),
        replay
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0))
    );
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Library));
    assert_eq!(zone_of(&s, target), Some(ZoneType::Graveyard));
    assert_ne!(s.game().state.rng.get_word_pos(), rng);
}

#[test]
fn copied_charring_bite_shuffles_before_the_original_loses_its_target() {
    use crate::support::{
        game::{force_lands, move_from_library},
        inspect::zone_of,
        play::choose,
    };
    use managym::{
        agent::action::Action,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s =
        five_color_unfunded_setup(&["Pyromancer's Goggles", "Twinmaw Stormbrood", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Pyromancer's Goggles");
    force_lands(&mut s, 0, "Mountain", 1);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let spell = move_from_library(&mut s, 0, "Twinmaw Stormbrood", ZoneType::Hand);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.game_mut().scenario_refresh_priority().unwrap();
    let size = s.zone_size(0, ZoneType::Library);
    let rng = s.game().state.rng.get_word_pos();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell {object, profile: SpellProfile::Omen, ..} if *object == spell),
    );
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard));
    s.assert_zone_size(0, ZoneType::Library, size);
    assert_ne!(s.game().state.rng.get_word_pos(), rng);
}

#[test]
fn free_exile_grant_casts_charring_bite_with_no_mana() {
    use crate::support::{game::move_from_library, inspect::zone_of, play::choose};
    use managym::{
        agent::action::Action,
        flow::cast_permission::PlayGrant,
        state::{card::SpellProfile, game_object::Target, mana::ManaCost},
    };
    let mut s = five_color_unfunded_setup(&["Twinmaw Stormbrood", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let spell = move_from_library(&mut s, 0, "Twinmaw Stormbrood", ZoneType::Exile);
    s.game_mut().state.cast_permissions.grant(PlayGrant {
        cost: Some(ManaCost::default()),
        ..PlayGrant::new(spell, PlayerId(0))
    });
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell {object, profile: SpellProfile::Omen, ..} if *object == spell),
    );
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert_eq!(top_spell(&s).spent.total(), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Library));
}

#[test]
fn scavenger_regent_exude_toxin_paid_and_free() {
    use crate::support::{
        game::move_from_library,
        inspect::zone_of,
        play::{choose, choose_mode},
    };
    use managym::{
        agent::{action::Action, observation::Observation},
        flow::cast_permission::PlayGrant,
        state::{card::SpellProfile, mana::ManaCost},
    };
    for (free, x) in [(false, 0), (false, 2), (true, 0)] {
        let mut s = five_color_unfunded_setup(&["Scavenger Regent", "Gray Ogre"]);
        let dragon = s.force_permanent_on_battlefield(1, "Scavenger Regent");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let ogre = card_of(&s, ogre);
        let spell = move_from_library(
            &mut s,
            0,
            "Scavenger Regent",
            if free {
                ZoneType::Exile
            } else {
                ZoneType::Hand
            },
        );
        if free {
            s.game_mut().state.cast_permissions.grant(PlayGrant {
                cost: Some(ManaCost::default()),
                ..PlayGrant::new(spell, PlayerId(0))
            });
        }
        s.set_player_mana_pool(0, "2BB");
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        s.game_mut().scenario_refresh_priority().unwrap();
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { object, profile: SpellProfile::Omen, .. } if *object == spell),
        );
        if !free {
            choose_mode(&mut s, x);
        }
        assert_eq!(
            top_spell(&s).spent.total() as usize,
            if free { 0 } else { 2 + x }
        );
        for viewer in [PlayerId(0), PlayerId(1)] {
            let observation = Observation::for_player(s.game(), viewer);
            let item = &observation.stack_objects[0];
            assert_eq!(item.spell.as_ref().unwrap().name, "Exude Toxin");
            assert_eq!(item.x, x as i32);
            assert_eq!(item.effective_mana_value(), Some(2 + x as u8));
        }
        let rng = s.game().state.rng.get_word_pos();
        settle(&mut s);
        assert_ne!(s.game().state.rng.get_word_pos(), rng);
        assert_eq!(zone_of(&s, spell), Some(ZoneType::Library));
        assert_eq!(
            zone_of(&s, ogre),
            Some(if x == 2 {
                ZoneType::Graveyard
            } else {
                ZoneType::Battlefield
            })
        );
        assert_eq!(s.game().characteristics(dragon).pt(), (4, 4));
    }
}

#[test]
fn scavenger_regent_ward_offers_each_discard_and_decline() {
    use crate::support::{
        game::move_from_library,
        inspect::{permanent, zone_of},
        play::{choose, fork_choosing},
    };
    use managym::{agent::action::Action, state::game_object::Target};
    let mut s = five_color_unfunded_setup(&["Scavenger Regent", "Lightning Bolt"]);
    let dragon = s.force_permanent_on_battlefield(1, "Scavenger Regent");
    let bolt = move_from_library(&mut s, 0, "Lightning Bolt", ZoneType::Hand);
    let cards =
        ["Mountain", "Forest"].map(|name| move_from_library(&mut s, 0, name, ZoneType::Hand));
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Permanent(dragon)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(offered_cards(&s).len(), 2);
    let mut declined = fork_choosing(&s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut declined);
    assert_eq!(permanent(&declined, dragon).damage, 0);
    for card in cards {
        assert!(offered_cards(&s).contains(&card));
    }
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard {object,..} if *object == cards[0]),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, cards[0]), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, cards[1]), Some(ZoneType::Hand));
    assert_eq!(permanent(&s, dragon).damage, 3);
}

#[test]
fn skimming_strike_draws_without_a_target_or_after_tapping_one() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{hand, permanent_of, tapped, zone_of},
        play::{cast_profile, choose},
    };
    use managym::{
        agent::action::Action,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s = five_color_setup(&["Dirgur Island Dragon", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = move_from_library(&mut s, 0, "Dirgur Island Dragon", ZoneType::Hand);
    let mut front = s.clone();
    cast_profile(&mut front, card, SpellProfile::Front);
    settle(&mut front);
    let object = card_of(&front, permanent_of(&front, card).unwrap());
    assert_eq!(
        front.game().state[object]
            .ward()
            .unwrap()
            .mana()
            .unwrap()
            .mana_value,
        2
    );
    cast_profile(&mut s, card, SpellProfile::Omen);
    for tap in [true, false] {
        let mut branch = s.clone();
        if tap {
            assert!(branch.choose_target(Target::Permanent(ogre)));
        } else {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut branch);
        assert_eq!(tapped(&branch, ogre), tap);
        assert_eq!(hand(&branch, 0).len(), 1);
        assert_eq!(zone_of(&branch, card), Some(ZoneType::Library));
    }
}

#[test]
fn roost_seek_finds_a_basic_and_sagu_wildling_gains_three() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{hand, zone_of},
        play::{cast_profile, resolve_top},
    };
    use managym::state::card::SpellProfile;
    let mut s = five_color_setup(&["Sagu Wildling"]);
    let card = move_from_library(&mut s, 0, "Sagu Wildling", ZoneType::Hand);
    let mut front = s.clone();
    cast_profile(&mut front, card, SpellProfile::Front);
    settle(&mut front);
    assert_eq!(front.life(0), 23);
    cast_profile(&mut s, card, SpellProfile::Omen);
    resolve_top(&mut s);
    let candidates = offered_cards(&s);
    assert!(!candidates.is_empty());
    assert!(candidates
        .iter()
        .all(|c| s.game().state[*c].types.is_land()));
    select_card(&mut s, candidates[0]);
    settle(&mut s);
    assert_eq!(hand(&s, 0), [now(&s, candidates[0])]);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Library));
}

#[test]
fn disruptive_stormbrood_and_petty_revenge_destroy_different_targets() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::zone_of,
        play::{cast_profile, pass_until},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s = five_color_setup(&[
        "Disruptive Stormbrood",
        "Omniscience",
        "Gray Ogre",
        "Craw Wurm",
    ]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let artifact_or_enchantment = s.force_permanent_on_battlefield(1, "Omniscience");
    let small = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let large = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let card = move_from_library(&mut s, 0, "Disruptive Stormbrood", ZoneType::Hand);
    for (profile, target) in [
        (SpellProfile::Omen, small),
        (SpellProfile::Front, artifact_or_enchantment),
    ] {
        let mut branch = s.clone();
        let object = card_of(&branch, target);
        cast_profile(&mut branch, card, profile);
        pass_until(&mut branch, ActionSpaceKind::ChooseTarget);
        assert!(!branch.choose_target(Target::Permanent(large)));
        assert!(branch.choose_target(Target::Permanent(target)));
        settle(&mut branch);
        assert_eq!(zone_of(&branch, object), Some(ZoneType::Graveyard));
    }
}

#[test]
fn dynamic_soar_receives_whirlwings_sorcery_timing_and_purging_removes_its_counters() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{counters, keywords, permanent_of},
        play::{cast_profile, pass_until},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        flow::turn::StepKind,
        state::{
            card::{Keyword, SpellProfile},
            game_object::Target,
        },
    };
    let mut s = five_color_setup(&["Whirlwing Stormbrood", "Purging Stormbrood", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let whirl = move_from_library(&mut s, 0, "Whirlwing Stormbrood", ZoneType::Hand);
    let second = move_from_library(&mut s, 0, "Whirlwing Stormbrood", ZoneType::Hand);
    let purge = move_from_library(&mut s, 0, "Purging Stormbrood", ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::Upkeep);
    s.set_player_mana_pool(0, "14UUGBWW");
    cast_profile(&mut s, whirl, SpellProfile::Front);
    settle(&mut s);
    cast_profile(&mut s, second, SpellProfile::Omen);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(counters(&s, ogre), 3);
    s.set_player_mana_pool(0, "4BW");
    let mut absorb = s.clone();
    cast_profile(&mut absorb, purge, SpellProfile::Omen);
    assert!(absorb.choose_target(Target::Permanent(ogre)));
    settle(&mut absorb);
    assert_eq!(absorb.game().characteristics(ogre).pt(), (7, 7));
    for keyword in [Keyword::Lifelink, Keyword::Hexproof] {
        assert!(keywords(&absorb, ogre).has(keyword));
    }
    cast_profile(&mut s, purge, SpellProfile::Front);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(counters(&s, ogre), 0);
    let object = card_of(&s, permanent_of(&s, purge).unwrap());
    assert_eq!(
        s.game().state[object].ward(),
        Some(&managym::state::ability::PaymentCost::Life(2))
    );
}

#[test]
fn riling_dawnbreaker_creates_a_white_soldier_or_pumps_another_creature() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::permanent_of,
        play::cast_profile,
    };
    use managym::{
        flow::turn::StepKind,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s = five_color_setup(&["Riling Dawnbreaker"]);
    let card = move_from_library(&mut s, 0, "Riling Dawnbreaker", ZoneType::Hand);
    let second = move_from_library(&mut s, 0, "Riling Dawnbreaker", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Omen);
    settle(&mut s);
    let [soldier] = s.battlefield_permanents_named(0, "Soldier")[..] else {
        panic!("one Soldier")
    };
    assert_eq!(s.game().characteristics(soldier).pt(), (2, 2));
    assert!(s.game().state[card_of(&s, soldier)]
        .colors()
        .contains(&Color::White));
    cast_profile(&mut s, second, SpellProfile::Front);
    settle(&mut s);
    let dragon = permanent_of(&s, second).unwrap();
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(!s.choose_target(Target::Permanent(dragon)));
    assert!(s.choose_target(Target::Permanent(soldier)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(soldier).pt(), (3, 2));
}

#[test]
fn marang_bounces_two_other_nonlands_or_draws_then_discards() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{hand, permanent_of, zone_of},
        play::{cast_profile, pass_until, resolve_top},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{card::SpellProfile, game_object::Target},
    };
    let mut s = five_color_setup(&["Marang River Regent", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let ogres = [0, 1].map(|p| s.force_permanent_on_battlefield(p, "Gray Ogre"));
    let land = s.force_permanent_on_battlefield(0, "Island");
    let card = move_from_library(&mut s, 0, "Marang River Regent", ZoneType::Hand);
    let mut front = s.clone();
    cast_profile(&mut front, card, SpellProfile::Front);
    pass_until(&mut front, ActionSpaceKind::ChooseTarget);
    let dragon = permanent_of(&front, card).unwrap();
    assert!(!front.choose_target(Target::Permanent(dragon)));
    assert!(!front.choose_target(Target::Permanent(land)));
    for ogre in ogres {
        assert!(front.choose_target(Target::Permanent(ogre)));
    }
    settle(&mut front);
    for ogre in ogres {
        assert_eq!(zone_of(&front, card_of(&s, ogre)), Some(ZoneType::Hand));
    }
    cast_profile(&mut s, card, SpellProfile::Omen);
    resolve_top(&mut s);
    assert_eq!(hand(&s, 0).len(), 3);
    let discard = offered_cards(&s)[0];
    select_card(&mut s, discard);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 2);
    assert_eq!(zone_of(&s, discard), Some(ZoneType::Graveyard));
}

#[test]
fn flush_out_draws_only_after_discard_and_stormshriek_pumps_itself() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{hand, permanent_of},
        play::{activate, cast_profile},
    };
    use managym::state::card::SpellProfile;
    let mut s = five_color_setup(&["Stormshriek Feral"]);
    let card = move_from_library(&mut s, 0, "Stormshriek Feral", ZoneType::Hand);
    let mut front = s.clone();
    cast_profile(&mut front, card, SpellProfile::Front);
    settle(&mut front);
    let dragon = permanent_of(&front, card).unwrap();
    let (power, toughness) = front.game().characteristics(dragon).pt();
    activate(&mut front, dragon, 0);
    settle(&mut front);
    assert_eq!(
        front.game().characteristics(dragon).pt(),
        (power + 1, toughness)
    );
    for available in [false, true] {
        let mut branch = s.clone();
        if available {
            move_from_library(&mut branch, 0, "Mountain", ZoneType::Hand);
        }
        cast_profile(&mut branch, card, SpellProfile::Omen);
        settle(&mut branch);
        assert_eq!(hand(&branch, 0).len(), if available { 2 } else { 0 });
    }
}

#[test]
fn runescale_triggers_for_noncreatures_and_dragons_and_chilling_screech_caps_mana_value() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{permanent_of, zone_of},
        play::cast_profile,
    };
    use managym::state::{card::SpellProfile, game_object::Target};
    let mut s = five_color_setup(&[
        "Runescale Stormbrood",
        "Gray Ogre",
        "Think Twice",
        "Sagu Wildling",
    ]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let card = move_from_library(&mut s, 0, "Runescale Stormbrood", ZoneType::Hand);
    let answer = move_from_library(&mut s, 0, "Runescale Stormbrood", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Front);
    settle(&mut s);
    let runescale = permanent_of(&s, card).unwrap();
    let (power, toughness) = s.game().characteristics(runescale).pt();
    for (target, bonus) in [("Gray Ogre", 0), ("Sagu Wildling", 2), ("Think Twice", 2)] {
        let mut branch = s.clone();
        let spell = move_from_library(&mut branch, 0, target, ZoneType::Hand);
        cast_profile(&mut branch, spell, SpellProfile::Front);
        // Triggered P/T changes resolve before the spell.
        if bonus != 0 {
            crate::support::play::resolve_top(&mut branch);
        }
        assert_eq!(
            branch.game().characteristics(runescale).pt(),
            (power + bonus, toughness)
        );
        let target_id = now(&branch, spell);
        let legal=branch.action_space().actions.iter().any(|a|matches!(a,managym::agent::action::Action::CastSpell{object,profile:SpellProfile::Omen,..} if *object==answer));
        assert_eq!(legal, target == "Think Twice");
        if legal {
            cast_profile(&mut branch, answer, SpellProfile::Omen);
            assert!(branch.choose_target(Target::StackSpell(target_id)));
            settle(&mut branch);
            assert_eq!(zone_of(&branch, spell), Some(ZoneType::Graveyard));
            assert_eq!(branch.zone_size(0, ZoneType::Hand), 0);
        }
    }
}

#[test]
fn feral_deathgorger_exiles_up_to_two_from_one_chosen_graveyard() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::{counters, hand, zone_of},
        play::{cast_profile, choose, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{card::SpellProfile, game_object::Target},
    };
    let mut root = five_color_setup(&["Feral Deathgorger", "Gray Ogre"]);
    let ogre = root.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = move_from_library(&mut root, 0, "Feral Deathgorger", ZoneType::Hand);
    root.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let graves = [0, 1].map(|p| {
        ["Island", "Forest"].map(|land| move_from_library(&mut root, p, land, ZoneType::Graveyard))
    });
    let mut entry = root.clone();
    cast_profile(&mut entry, card, SpellProfile::Front);
    pass_until(&mut entry, ActionSpaceKind::ChooseTarget);
    let decline = |a: &Action| matches!(a, Action::Decline { .. });
    for seat in [0, 1] {
        for count in 0..=2 {
            let mut branch = entry.clone();
            if count > 0 {
                assert!(branch.choose_target(Target::GraveyardCard(graves[seat][0])));
                assert!(!branch.choose_target(Target::GraveyardCard(graves[1 - seat][0])));
            }
            if count == 2 {
                assert!(branch.choose_target(Target::GraveyardCard(graves[seat][1])));
            } else {
                choose(&mut branch, decline);
            }
            settle(&mut branch);
            for (i, id) in graves[seat].iter().enumerate() {
                assert_eq!(
                    zone_of(&branch, *id),
                    Some(if i < count {
                        ZoneType::Exile
                    } else {
                        ZoneType::Graveyard
                    })
                );
            }
        }
    }
    cast_profile(&mut root, card, SpellProfile::Omen);
    for targeted in [true, false] {
        let mut branch = root.clone();
        if targeted {
            assert!(branch.choose_target(Target::Permanent(ogre)));
        } else {
            choose(&mut branch, decline);
        }
        settle(&mut branch);
        assert_eq!(counters(&branch, ogre), i32::from(targeted));
        assert_eq!(hand(&branch, 0).len(), 1);
        assert_eq!(zone_of(&branch, card), Some(ZoneType::Library));
    }
}

/// Claim Territory finds up to two basic Forests, then its caster chooses
/// which one enters tapped; the other goes to hand, and the Omen shuffles
/// into the library. Bloomvine Regent gains 3 life as it or another Dragon
/// enters.
#[test]
fn bloomvine_regent_gains_life_for_dragons_and_claim_territory_splits_two_forests() {
    use crate::support::{
        game::{five_color_setup, move_from_library},
        inspect::hand,
        play::cast_profile,
    };
    use managym::state::card::SpellProfile;
    let mut s = five_color_setup(&["Bloomvine Regent", "Twinmaw Stormbrood"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let regent = move_from_library(&mut s, 0, "Bloomvine Regent", ZoneType::Hand);
    s.set_player_mana_pool(0, "2G");
    cast_profile(&mut s, regent, SpellProfile::Omen);
    resolve_search(&mut s);
    let forests = offered_cards(&s);
    assert!(forests.len() >= 2);
    assert!(forests
        .iter()
        .all(|id| s.game().state[*id].name == "Forest"));
    select_card(&mut s, forests[0]);
    select_card(&mut s, forests[1]);
    // Both found; now choose the one that enters tapped.
    let found = offered_cards(&s);
    assert_eq!(found.len(), 2, "{:?}", s.action_space());
    select_card(&mut s, found[1]);
    settle(&mut s);
    let battlefield = s.battlefield_permanents_named(0, "Forest");
    assert_eq!(battlefield.len(), 1);
    assert!(
        s.game().state.permanents[battlefield[0]]
            .as_ref()
            .unwrap()
            .tapped
    );
    assert_eq!(
        s.game().state.permanent_of(now(&s, found[1])),
        Some(battlefield[0])
    );
    assert_eq!(hand(&s, 0), [now(&s, found[0])]);
    assert_eq!(
        s.game().state.zones.zone_of(now(&s, regent)),
        Some(ZoneType::Library),
        "the Omen shuffles itself"
    );

    let mut s = five_color_setup(&["Bloomvine Regent", "Twinmaw Stormbrood"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let life = s.life(0);
    let regent = move_from_library(&mut s, 0, "Bloomvine Regent", ZoneType::Hand);
    cast_profile(&mut s, regent, SpellProfile::Front);
    settle(&mut s);
    assert_eq!(s.life(0), life + 3);
    let dragon = move_from_library(&mut s, 0, "Twinmaw Stormbrood", ZoneType::Hand);
    cast_profile(&mut s, dragon, SpellProfile::Front);
    settle(&mut s);
    // Twinmaw's own enter trigger gains 5 more.
    assert_eq!(s.life(0), life + 3 + 3 + 5);
}

fn resolve_search(s: &mut crate::support::game::Scenario) {
    s.pass_priority();
    s.pass_priority();
}
