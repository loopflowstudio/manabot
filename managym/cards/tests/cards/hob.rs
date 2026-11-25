//! Beorn's real combat trigger composes with Kellan's independently resolved upgrades.
use crate::support::{
    game::five_color_setup,
    inspect::permanent_named,
    play::{activate, enter_and_settle, pass_until, settle},
};
use managym::state::card::SpellProfile;
use managym::{
    agent::{
        action::ActionSpaceKind, observation::Observation,
        observation_encoder::ObservationEncoderConfig, observation_rows::InputRows,
    },
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{PlayerId, Target},
        permanent::CounterKind,
    },
};

#[test]
fn beorn_adds_bear_and_keeps_trample_after_kellan_replaces_types() {
    for upgrade_first in [false, true] {
        let mut s = five_color_setup(&["Beorn the Fierce", "Kellan, Planar Trailblazer"]);
        enter_and_settle(&mut s, "Beorn the Fierce");
        enter_and_settle(&mut s, "Kellan, Planar Trailblazer");
        let k = permanent_named(&s, "Kellan, Planar Trailblazer");
        activate(&mut s, k, 0);
        settle(&mut s);
        if upgrade_first {
            activate(&mut s, k, 1);
            settle(&mut s);
        }
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(k)));
        settle(&mut s);
        if !upgrade_first {
            s.set_player_mana_pool(0, "2R");
            activate(&mut s, k, 1);
            settle(&mut s);
        }
        let view = s.game().characteristics(k);
        assert_eq!(view.has_subtype("Bear"), upgrade_first);
        assert_eq!(view.pt(), if upgrade_first { (5, 4) } else { (3, 2) });
        assert!(view.keywords.has(Keyword::Trample) && view.keywords.has(Keyword::DoubleStrike));
        assert_eq!(view.triggered_abilities.len(), 1);
        assert_eq!(
            s.game().state.permanents[k]
                .as_ref()
                .unwrap()
                .counter_count(CounterKind::Trample),
            1
        );
        // Clone an actual priority boundary, preserving input and pending offers.
        let fork = s.game().clone();
        let obs = Observation::for_player(s.game(), PlayerId(0));
        let config = ObservationEncoderConfig::default();
        assert_eq!(
            InputRows::read(&obs, &config).unwrap(),
            InputRows::read(&Observation::for_player(&fork, PlayerId(0)), &config).unwrap()
        );
    }
}

#[test]
fn beorn_draws_with_three_bears_and_renews_a_keyword_counter_after_loss() {
    use managym::state::{
        effect::{EffectDuration, Modification},
        zone::ZoneType,
    };
    let mut s = five_color_setup(&["Beorn the Fierce", "Platypus-Bear", "Gray Ogre"]);
    enter_and_settle(&mut s, "Beorn the Fierce");
    let bear = s.force_permanent_on_battlefield(0, "Platypus-Bear");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand + 2
    );
    assert_eq!(s.game().characteristics(bear).pt(), (4, 5));
    let first = s.game().state.permanents[ogre]
        .as_ref()
        .unwrap()
        .keyword_counter_timestamps[&CounterKind::Trample];
    let affected = s.game().affected_permanent(ogre).unwrap();
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected],
        Modification::LoseAllAbilities,
        EffectDuration::Indefinite,
    );
    assert!(!s
        .game()
        .characteristics(ogre)
        .keywords
        .has(Keyword::Trample));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let p = s.game().state.permanents[ogre].as_ref().unwrap();
    assert_eq!(p.counter_count(CounterKind::Trample), 2);
    assert!(p.keyword_counter_timestamps[&CounterKind::Trample] > first);
    assert!(s
        .game()
        .characteristics(ogre)
        .keywords
        .has(Keyword::Trample));
}

#[test]
fn bilbo_graveyard_adventure_replacement_owns_the_exile_permission() {
    use crate::support::{
        game::{force_lands, move_card, move_from_library},
        inspect::{can_cast, card_of, now, offer_labels, zone_of},
        play::{attack_with, choose, fork_choosing, fork_stepping},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&[
        "Bilbo, Thief in the Night",
        "Smaug, the Great Calamity",
        "Gray Ogre",
        "Counterspell",
        "Lightning Bolt",
    ]);
    let bilbo = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Bilbo, Thief in the Night", true)
        .unwrap();
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let target = card_of(&s, ogre);
    let smaug = move_from_library(&mut s, 0, "Smaug, the Great Calamity", ZoneType::Graveyard);
    let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
    let bolt = move_from_library(&mut s, 1, "Lightning Bolt", ZoneType::Hand);
    force_lands(&mut s, 1, "Mountain", 1);
    // Exactly four mana pays for the reduced Spew Flame.
    force_lands(&mut s, 0, "Mountain", 4);
    force_lands(&mut s, 1, "Island", 2);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, bilbo);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell {profile: SpellProfile::Adventure, object,..} if *object==smaug),
    );
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    assert_eq!(crate::support::inspect::top_spell(&s).spent.mana().len(), 4);
    let mut lost_target = fork_choosing(
        &s,
        |a| matches!(a, Action::CastSpell {object,..} if *object == bolt),
    );
    assert!(lost_target.choose_target(Target::Permanent(ogre)));
    settle(&mut lost_target);
    assert_eq!(zone_of(&lost_target, target), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&lost_target, smaug), Some(ZoneType::Exile));
    let mut countered = fork_choosing(
        &s,
        |a| matches!(a,Action::CastSpell {object,..} if *object == counter),
    );
    let spell = now(&countered, smaug);
    assert!(countered.choose_target(Target::StackSpell(spell)));
    settle(&mut countered);
    assert_eq!(zone_of(&countered, smaug), Some(ZoneType::Exile));
    assert_eq!(zone_of(&countered, target), Some(ZoneType::Battlefield));

    for branch in [&mut lost_target, &mut countered] {
        branch.advance_to_active_step(0, StepKind::PostcombatMain);
        branch.set_player_mana_pool(0, "5RR");
        branch.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(branch, smaug));
    }

    s.pass_priority();
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(
        offer_labels(&s),
        [
            "Spew Flame: Exile on Adventure",
            "Spew Flame: Exile without Adventure permission"
        ]
    );
    for (index, label) in offer_labels(&s).iter().enumerate() {
        let grants = label.contains("on Adventure");
        let mut branch = fork_stepping(&s, index);
        settle(&mut branch);
        assert_eq!(zone_of(&branch, target), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&branch, smaug), Some(ZoneType::Exile));
        branch.advance_to_active_step(0, StepKind::PostcombatMain);
        branch.set_player_mana_pool(0, "5RR");
        branch.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&branch, smaug), grants);
        move_card(&mut branch, smaug, ZoneType::Graveyard);
        move_card(&mut branch, smaug, ZoneType::Exile);
        branch.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_cast(&branch, smaug));
    }
}

/// A paid resolution offer previews its payment whichever profile it casts:
/// the front-face Bolt (a card selection) and the Adventure Spew Flame.
#[test]
fn bilbo_previews_front_face_and_adventure_graveyard_casts_alike() {
    use crate::support::{
        game::{force_lands, move_from_library},
        play::attack_with,
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&[
        "Bilbo, Thief in the Night",
        "Smaug, the Great Calamity",
        "Lightning Bolt",
    ]);
    let bilbo = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Bilbo, Thief in the Night", true)
        .unwrap();
    let smaug = move_from_library(&mut s, 0, "Smaug, the Great Calamity", ZoneType::Graveyard);
    let bolt = move_from_library(&mut s, 0, "Lightning Bolt", ZoneType::Graveyard);
    force_lands(&mut s, 0, "Mountain", 4);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, bilbo);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let frame = s.game().semantic_decision_frame().unwrap();
    let preview = |offered: &dyn Fn(&Action) -> bool| {
        let index = s.action_space().actions.iter().position(offered).unwrap();
        let preview = s
            .game()
            .payment_preview(PlayerId(0), frame.revision, &frame.fingerprint, index)
            .unwrap();
        serde_json::to_value(&preview).unwrap()
    };
    let front = preview(&|a| matches!(a, Action::SelectCard { object, .. } if *object == bolt));
    let adventure = preview(
        &|a| matches!(a, Action::CastSpell { object, profile: SpellProfile::Adventure, .. } if *object == smaug),
    );
    assert_eq!(front["state"], adventure["state"], "{front} {adventure}");
    assert_eq!(front["cost"], "{R}", "{front}");
    assert!(adventure["cost"].is_string(), "{adventure}");
    assert_eq!(front["sources"].as_array().unwrap().len(), 1, "{front}");
}

#[test]
fn spew_flame_cost_reductions_read_the_sorcery_profile() {
    use crate::support::{game::move_from_library, inspect::can_cast};
    use managym::state::zone::ZoneType;
    let name = "Smaug, the Great Calamity";
    for (reducer, eligible) in [("Mocking Sprite", true), ("Dragonlord's Servant", false)] {
        let mut game = five_color_setup(&[name, reducer]);
        game.force_permanent_on_battlefield(0, reducer);
        let smaug = move_from_library(&mut game, 0, name, ZoneType::Hand);
        game.set_player_mana_pool(0, "3R");
        game.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&game, smaug), eligible, "{reducer}");
    }
}

#[test]
fn overlapping_grants_price_each_smaug_profile() {
    use crate::support::game::{five_color_unfunded_setup, move_from_library};
    use managym::{
        flow::cast_permission::PlayGrant,
        state::{mana::ManaCost, zone::ZoneType},
    };

    let name = "Smaug, the Great Calamity";
    let mut s = five_color_unfunded_setup(&[name, "Gray Ogre"]);
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = move_from_library(&mut s, 0, name, ZoneType::Exile);
    for cost in [None, Some(ManaCost::default())] {
        s.game_mut().state.cast_permissions.grant(PlayGrant {
            cost,
            ..PlayGrant::new(card, PlayerId(0))
        });
    }
    s.set_player_mana_pool(0, "5RR");
    s.game_mut().scenario_refresh_priority().unwrap();
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let mut prices: Vec<_> = obs
        .action_space
        .actions
        .iter()
        .filter_map(|offer| Some((offer.spell_profile, offer.route_mana_value?)))
        .collect();
    prices.sort_unstable();
    assert_eq!(prices, [(0, 0), (0, 7), (1, 0), (1, 5)]);
}

/// Mid-announcement projections name the selected Adventure, not its card,
/// and keep the private announcement from the opponent.
#[test]
fn spew_flame_announcement_names_the_adventure() {
    use crate::support::{game::move_from_library, play::choose};
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&["Smaug, the Great Calamity", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let smaug = move_from_library(&mut s, 0, "Smaug, the Great Calamity", ZoneType::Hand);
    s.set_player_mana_pool(0, "4R");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { object, profile: SpellProfile::Adventure, .. } if *object == smaug),
    );
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "targets");
    assert_eq!(context.announcing.as_deref(), Some("spell"));
    assert_eq!(context.source.0.unwrap().label, "Spew Flame");
    let frames = s.game().continuation_frames(PlayerId(0));
    let announcement = frames.iter().find(|frame| frame.kind == 4).unwrap();
    assert_eq!(announcement.spell_profile, 1);
    assert!(s
        .game()
        .decision_context(PlayerId(1))
        .unwrap()
        .source
        .0
        .is_none());
    assert!(s
        .game()
        .continuation_frames(PlayerId(1))
        .iter()
        .all(|frame| frame.spell_profile == 0));
    assert!(s.choose_target(Target::Permanent(ogre)));
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(
        obs.stack_objects[0].spell.as_ref().unwrap().name,
        "Spew Flame"
    );
}

#[test]
fn compound_spew_flame_keeps_its_profile_and_target() {
    use crate::support::{
        game::move_from_library,
        inspect::{top_spell, zone_of},
    };
    use managym::{
        agent::structured_offer::{ChoiceAnswer, ChoiceStep, OfferSubmission},
        state::zone::ZoneType,
    };
    let mut s = five_color_setup(&["Smaug, the Great Calamity", "Gray Ogre", "Hungry Ghoul"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Hungry Ghoul");
    let smaug = move_from_library(&mut s, 0, "Smaug, the Great Calamity", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    let offers = s.game().compound_offers().unwrap();
    let front = offers
        .projection()
        .offers
        .iter()
        .find(|offer| matches!(&offer.public_commitment, Some(managym::agent::structured_offer::PublicCommitment::Cast { card }) if card == "Smaug, the Great Calamity"))
        .unwrap();
    assert!(front.choices.is_empty());
    let inset = offers
        .projection()
        .offers
        .iter()
        .find(|offer| offer.label.contains("Spew Flame"))
        .unwrap();
    let [ChoiceStep::Select {
        role, candidates, ..
    }] = &inset.choices[..]
    else {
        panic!("Adventure target")
    };
    let target = candidates
        .initial
        .as_ref()
        .unwrap()
        .iter()
        .find(|target| target.label == "Gray Ogre")
        .unwrap();
    let submission = OfferSubmission {
        offer_id: inset.id,
        answers: vec![ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![target.id],
        }],
    };
    assert_eq!(
        s.game()
            .compound_commands(&offers, &submission)
            .unwrap()
            .len(),
        2
    );
    s.game_mut()
        .apply_offer_submission(&offers, &submission)
        .unwrap();
    assert_eq!(top_spell(&s).profile, SpellProfile::Adventure);
    assert_eq!(top_spell(&s).targets, [Target::Permanent(ogre)]);
    settle(&mut s);
    assert_eq!(zone_of(&s, smaug), Some(ZoneType::Exile));
}

#[test]
fn concerted_care_protects_a_controlled_target_then_bofur_has_lifelink() {
    use crate::support::{
        game::move_from_library,
        inspect::{keywords, permanent_of},
        play::cast_profile,
    };
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Bofur, Reliable Guardian", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let bofur = move_from_library(&mut s, 0, "Bofur, Reliable Guardian", ZoneType::Hand);
    cast_profile(&mut s, bofur, SpellProfile::Adventure);
    assert!(!s.choose_target(Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ally)));
    settle(&mut s);
    for keyword in [Keyword::Hexproof, Keyword::Indestructible] {
        assert!(keywords(&s, ally).has(keyword));
    }
    cast_profile(&mut s, bofur, SpellProfile::Front);
    settle(&mut s);
    assert!(keywords(&s, permanent_of(&s, bofur).unwrap()).has(Keyword::Lifelink));
}

#[test]
fn gaze_in_wonder_accepts_one_or_two_distinct_creatures() {
    use crate::support::{
        game::move_from_library,
        inspect::{keywords, permanent_of, tapped},
        play::{cast_profile, choose},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&["Velvetwing Butterflies", "Gray Ogre"]);
    let ogres = [0, 1].map(|seat| s.force_permanent_on_battlefield(seat, "Gray Ogre"));
    let card = move_from_library(&mut s, 0, "Velvetwing Butterflies", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Adventure);
    assert!(s.choose_target(Target::Permanent(ogres[0])));
    for both in [false, true] {
        let mut branch = s.clone();
        assert!(!branch.choose_target(Target::Permanent(ogres[0])));
        if both {
            assert!(branch.choose_target(Target::Permanent(ogres[1])));
        } else {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut branch);
        assert!(tapped(&branch, ogres[0]));
        assert_eq!(tapped(&branch, ogres[1]), both);
        cast_profile(&mut branch, card, SpellProfile::Front);
        settle(&mut branch);
        assert!(keywords(&branch, permanent_of(&branch, card).unwrap()).has(Keyword::Flying));
    }
}

#[test]
fn take_a_glance_scries_two_before_bilbo_draws_on_entry() {
    use crate::support::{
        game::move_from_library,
        inspect::hand,
        play::{cast_profile, resolve_top},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&["Bilbo Baggins, Burglar"]);
    let card = move_from_library(&mut s, 0, "Bilbo Baggins, Burglar", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Adventure);
    resolve_top(&mut s);
    let seen: std::collections::HashSet<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ScryCard { object, .. } => Some(*object),
            _ => None,
        })
        .collect();
    assert_eq!(seen.len(), 2);
    settle(&mut s);
    cast_profile(&mut s, card, SpellProfile::Front);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
}

#[test]
fn gone_fishing_returns_both_controlled_objects_as_new_appearances() {
    use crate::support::{
        game::move_from_library,
        inspect::{card_of, permanent_of},
        play::cast_profile,
    };
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Lake-town Mariners", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Island");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let old = [ogre, land].map(|p| card_of(&s, p));
    let card = move_from_library(&mut s, 0, "Lake-town Mariners", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Adventure);
    assert!(!s.choose_target(Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert!(!s.choose_target(Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    for id in old {
        assert_ne!(card_of(&s, permanent_of(&s, id).unwrap()), id);
    }
    cast_profile(&mut s, card, SpellProfile::Front);
    settle(&mut s);
    let id = permanent_of(&s, card).unwrap();
    let c = &s.game().state[card_of(&s, id)];
    assert!(c.keywords.has(Keyword::Vigilance));
    assert_eq!(c.ward().unwrap().mana().unwrap().mana_value, 2);
}

#[test]
fn meager_meal_separates_the_optional_creature_and_required_player() {
    use crate::support::{
        game::move_from_library,
        inspect::{counters, keywords, permanent_of},
        play::{cast_profile, choose},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut root = five_color_setup(&["Gollum, Silent Slinker", "Gray Ogre"]);
    let ogre = root.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = move_from_library(&mut root, 0, "Gollum, Silent Slinker", ZoneType::Hand);
    cast_profile(&mut root, card, SpellProfile::Adventure);
    for with_counter in [false, true] {
        let mut s = root.clone();
        if with_counter {
            assert!(s.choose_target(Target::Permanent(ogre)));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        settle(&mut s);
        assert_eq!(counters(&s, ogre), i32::from(with_counter));
        assert_eq!(s.life(1), 22);
        cast_profile(&mut s, card, SpellProfile::Front);
        settle(&mut s);
        assert!(keywords(&s, permanent_of(&s, card).unwrap()).has(Keyword::Menace));
    }
}

#[test]
fn easy_pickings_hits_opposing_creatures_and_gloin_adds_first_main_mana() {
    use crate::support::{
        game::move_from_library,
        inspect::{card_of, zone_of},
        play::cast_profile,
    };
    use managym::{
        flow::turn::StepKind,
        state::{mana::Mana, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Glóin the Mighty", "Savannah Lions"]);
    let own = s.force_permanent_on_battlefield(0, "Savannah Lions");
    let theirs = s.force_permanent_on_battlefield(1, "Savannah Lions");
    let own = card_of(&s, own);
    let theirs = card_of(&s, theirs);
    let card = move_from_library(&mut s, 0, "Glóin the Mighty", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Adventure);
    settle(&mut s);
    assert_eq!(zone_of(&s, own), Some(ZoneType::Battlefield));
    assert_eq!(zone_of(&s, theirs), Some(ZoneType::Graveyard));
    cast_profile(&mut s, card, SpellProfile::Front);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().state.players[1].mana_pool.total(), 0);
    s.advance_to_active_step(0, StepKind::Main);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("RR"));
}

#[test]
fn arkenstone_searches_for_a_legend_then_pumps_and_draws() {
    use crate::support::{
        game::move_from_library,
        inspect::{hand, offered_cards},
        play::{cast_profile, resolve_top, select_card},
    };
    use managym::{flow::turn::StepKind, state::zone::ZoneType};
    let mut s = five_color_setup(&["The Arkenstone", "Bilbo Baggins, Burglar", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = move_from_library(&mut s, 0, "The Arkenstone", ZoneType::Hand);
    cast_profile(&mut s, card, SpellProfile::Adventure);
    resolve_top(&mut s);
    let choices = offered_cards(&s);
    assert!(!choices.is_empty());
    assert!(choices
        .iter()
        .all(|&id| s.game().state[id].name == "Bilbo Baggins, Burglar"));
    select_card(&mut s, choices[0]);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
    cast_profile(&mut s, card, SpellProfile::Front);
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (3, 3));
    assert_eq!(s.game().characteristics(enemy).pt(), (2, 2));
    s.advance_to_active_step(0, StepKind::End);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 2);
}

#[test]
fn till_and_tend_stacks_land_plays_for_its_caster_until_cleanup() {
    use crate::support::{
        game::move_from_library,
        inspect::{keywords, permanent_of},
        play::{cast_profile, choose},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&["Beorn, Reluctant Host"]);
    let lands: Vec<_> = (0..4)
        .map(|_| move_from_library(&mut s, 0, "Forest", ZoneType::Hand))
        .collect();
    let first = move_from_library(&mut s, 0, "Beorn, Reluctant Host", ZoneType::Hand);
    let second = move_from_library(&mut s, 0, "Beorn, Reluctant Host", ZoneType::Hand);
    for card in [first, second] {
        cast_profile(&mut s, card, SpellProfile::Adventure);
        settle(&mut s);
    }
    for land in &lands[..3] {
        choose(
            &mut s,
            |a| matches!(a,Action::PlayLand{object,..} if object==land),
        );
    }
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::PlayLand { .. })));
    cast_profile(&mut s, first, SpellProfile::Front);
    settle(&mut s);
    assert!(keywords(&s, permanent_of(&s, first).unwrap()).has(Keyword::Trample));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(
        s.game()
            .player_effects(PlayerId(0))
            .filter(|effect| effect.modification
                == managym::state::effect::Modification::AdditionalLandPlay)
            .count(),
        0
    );
}

#[test]
fn speak_secrets_must_take_a_milled_spell_and_the_bird_tracks_threshold() {
    use crate::support::{
        game::{move_card, move_from_library, pin_top},
        inspect::{hand, now, offered_cards, permanent_of},
        play::{cast_profile, resolve_top, select_card},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&[
        "Most Decrepit Old Bird",
        "Think Twice",
        "Counterspell",
        "Gray Ogre",
    ]);
    let bird = move_from_library(&mut s, 0, "Most Decrepit Old Bird", ZoneType::Hand);
    let mill = pin_top(
        &mut s,
        &["Island", "Think Twice", "Counterspell", "Gray Ogre"],
    );
    s.set_player_mana_pool(0, "1UU");
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    cast_profile(&mut s, bird, SpellProfile::Adventure);
    resolve_top(&mut s);
    let choices = offered_cards(&s);
    assert_eq!(choices.len(), 2);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    for index in [1, 2] {
        let mut branch = s.clone();
        let chosen = now(&branch, mill[index]);
        assert!(choices.contains(&chosen));
        select_card(&mut branch, chosen);
        settle(&mut branch);
        assert_eq!(hand(&branch, 0), [now(&branch, chosen)]);
    }
    select_card(&mut s, choices[0]);
    settle(&mut s);
    s.set_player_mana_pool(0, "U");
    cast_profile(&mut s, bird, SpellProfile::Front);
    settle(&mut s);
    let permanent = permanent_of(&s, bird).unwrap();
    assert_eq!(s.game().characteristics(permanent).pt(), (1, 1));
    for _ in 0..3 {
        move_from_library(&mut s, 0, "Forest", ZoneType::Graveyard);
    }
    assert_eq!(s.game().characteristics(permanent).pt(), (1, 1));
    let seventh = move_from_library(&mut s, 0, "Forest", ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(permanent).pt(), (2, 2));
    move_card(&mut s, seventh, ZoneType::Exile);
    assert_eq!(s.game().characteristics(permanent).pt(), (1, 1));
}

#[test]
fn silvan_rally_takes_up_to_two_milled_lands_and_thranduil_buffs_the_elf() {
    use crate::support::{
        game::{move_from_library, pin_top},
        inspect::{hand, offered_cards, permanent_of},
        play::{cast_profile, choose, play_land, resolve_top, select_card},
    };
    use managym::{agent::action::Action, state::zone::ZoneType};
    let mut s = five_color_setup(&["Thranduil, Sindarin Liege", "Gray Ogre"]);
    let ruler = move_from_library(&mut s, 0, "Thranduil, Sindarin Liege", ZoneType::Hand);
    pin_top(&mut s, &["Gray Ogre", "Mountain", "Forest", "Island"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    cast_profile(&mut s, ruler, SpellProfile::Adventure);
    resolve_top(&mut s);
    let choices = offered_cards(&s);
    assert_eq!(choices.len(), 3);
    for count in 0..=2 {
        let mut branch = s.clone();
        for id in choices.iter().take(count) {
            select_card(&mut branch, *id);
        }
        if count < 2 {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut branch);
        assert_eq!(hand(&branch, 0).len(), count);
        cast_profile(&mut branch, ruler, SpellProfile::Front);
        settle(&mut branch);
        play_land(&mut branch, "Forest");
        settle(&mut branch);
        let elf = branch.battlefield_permanents_named(0, "Elf")[0];
        assert_eq!(branch.game().characteristics(elf).pt(), (2, 2));
        assert_eq!(
            branch
                .game()
                .characteristics(permanent_of(&branch, ruler).unwrap())
                .pt(),
            (2, 3)
        );
    }
}

#[test]
fn allure_sacrifices_before_drawing_and_my_precious_charges_both_equip_costs() {
    use crate::support::{
        game::move_from_library,
        inspect::{card_of, hand, keywords, permanent_of, zone_of},
        play::{cast_profile, select_card},
    };
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["My Precious", "Gray Ogre"]);
    let equipment = move_from_library(&mut s, 0, "My Precious", ZoneType::Hand);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let sacrifice = card_of(&s, first);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    cast_profile(&mut s, equipment, SpellProfile::Adventure);
    select_card(&mut s, sacrifice);
    assert_eq!(zone_of(&s, sacrifice), Some(ZoneType::Graveyard));
    assert!(hand(&s, 0).is_empty());
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 2);
    cast_profile(&mut s, equipment, SpellProfile::Front);
    settle(&mut s);
    let id = permanent_of(&s, equipment).unwrap();
    s.set_player_mana_pool(0, "2");
    activate(&mut s, id, 0);
    assert!(s.choose_target(Target::Permanent(second)));
    assert_eq!(s.life(0), 18);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert!(keywords(&s, second).has(Keyword::Hexproof));
    assert!(Observation::for_player(s.game(), PlayerId(0))
        .agent_permanents
        .iter()
        .any(|p| p.cant_be_blocked_this_turn));
}

/// Glamdring takes {X} off instant and sorcery spells, X the equipped
/// creature's power: Spew Flame costs {2}{R} with a 2/2 equipped and its
/// printed {4}{R} once that creature is gone. Gleam of Death mills six and
/// returns every instant and sorcery among them without a choice, then
/// goes on an Adventure.
#[test]
fn glamdring_discounts_by_equipped_power_and_gleam_returns_every_milled_spell() {
    use crate::support::{
        game::{move_card, move_from_library, pin_top},
        inspect::{hand, now, zone_of},
        play::{activate, cast_profile, castable},
    };
    use managym::state::{player::ManaPayment, zone::ZoneType};
    let mut s = five_color_setup(&[
        "Glamdring, Foe-hammer",
        "Smaug, the Great Calamity",
        "Gray Ogre",
        "Think Twice",
        "Counterspell",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let glamdring = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Glamdring, Foe-hammer", true)
        .unwrap();
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let smaug = move_from_library(&mut s, 0, "Smaug, the Great Calamity", ZoneType::Hand);
    let spew_flame_castable = |s: &mut crate::support::game::Scenario| {
        s.game_mut().scenario_refresh_priority().unwrap();
        let object = now(s, smaug);
        s.action_space().actions.iter().any(|a| {
            matches!(a, managym::agent::action::Action::CastSpell { object: o, profile: SpellProfile::Adventure, .. } if *o == object)
        })
    };
    s.set_player_mana_pool(0, "2R");
    assert!(!spew_flame_castable(&mut s), "unequipped: {{4}}{{R}}");

    s.set_player_mana_pool(0, "2");
    activate(&mut s, glamdring, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    s.set_player_mana_pool(0, "2R");
    assert!(spew_flame_castable(&mut s), "a 2/2 equipped: {{2}}{{R}}");
    let mut unequipped = s.clone();
    let creature = unequipped.game().state.permanents[ogre]
        .as_ref()
        .unwrap()
        .object;
    move_card(&mut unequipped, creature, ZoneType::Graveyard);
    assert!(!spew_flame_castable(&mut unequipped), "nothing equipped");
    cast_profile(&mut s, smaug, SpellProfile::Adventure);
    assert!(s.choose_target(Target::Permanent(enemy)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);

    // Gleam of Death: two of the six milled cards are instants or sorceries.
    let mut gleam = five_color_setup(&["Glamdring, Foe-hammer", "Think Twice", "Counterspell"]);
    gleam.game_mut().mana_payment[0] = ManaPayment::Auto;
    let card = move_from_library(&mut gleam, 0, "Glamdring, Foe-hammer", ZoneType::Hand);
    let milled = pin_top(
        &mut gleam,
        &[
            "Island",
            "Think Twice",
            "Forest",
            "Counterspell",
            "Swamp",
            "Plains",
        ],
    );
    gleam.set_player_mana_pool(0, "3U");
    assert!(castable(&mut gleam, card));
    cast_profile(&mut gleam, card, SpellProfile::Adventure);
    settle(&mut gleam);
    let mut returned: Vec<_> = hand(&gleam, 0)
        .into_iter()
        .map(|id| gleam.game().state[id].name.clone())
        .collect();
    returned.sort();
    assert_eq!(returned, ["Counterspell", "Think Twice"]);
    for land in [milled[0], milled[2], milled[4], milled[5]] {
        assert_eq!(zone_of(&gleam, land), Some(ZoneType::Graveyard));
    }
    assert_eq!(
        zone_of(&gleam, card),
        Some(ZoneType::Exile),
        "on an Adventure"
    );
}

/// Whether `viewer`'s observation shows the card behind `object`, and with
/// which face-down and foretold marks.
fn shown_to(
    s: &crate::support::game::Scenario,
    viewer: usize,
    object: managym::state::game_object::ObjectId,
) -> Option<(bool, bool)> {
    let id = s.game().state[object].id.0 as i32;
    let observation = Observation::for_player(s.game(), PlayerId(viewer));
    observation
        .agent_cards
        .iter()
        .chain(&observation.opponent_cards)
        .find(|card| card.id == id)
        .map(|card| {
            (
                card.face_down,
                card.cast_permissions.iter().any(|grant| grant.foretold),
            )
        })
}

/// Flameshape exiles the top two cards face down. Without a Wizard nobody,
/// its caster included, may look at or play them: they stay unseen, and
/// determinization resamples them. Gandalf shows them to his controller
/// alone and lets them be played; they hide again when the last Wizard
/// leaves and return with the next. Each noncreature spell grows Gandalf
/// and pings the opponent.
#[test]
fn flameshape_cards_are_hidden_and_locked_without_a_wizard() {
    use crate::support::{
        game::{move_card, move_from_library, pin_top},
        inspect::{can_cast, has_action, now, permanent_named, zone_of},
        play::{cast_card, cast_profile, choose},
    };
    use managym::{
        agent::action::Action,
        possible_worlds::HiddenHandConstraints,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&[
        "Gandalf, Goblins' Bane",
        "Lightning Strike",
        "High Fae Trickster",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let gandalf = move_from_library(&mut s, 0, "Gandalf, Goblins' Bane", ZoneType::Hand);
    let top = pin_top(&mut s, &["Lightning Strike", "Mountain"]);
    cast_profile(&mut s, gandalf, SpellProfile::Adventure);
    settle(&mut s);
    let (strike, mountain) = (now(&s, top[0]), now(&s, top[1]));
    assert_eq!(zone_of(&s, strike), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, mountain), Some(ZoneType::Exile));
    let playable = |s: &mut crate::support::game::Scenario| {
        s.game_mut().scenario_refresh_priority().unwrap();
        let (strike, mountain) = (now(s, strike), now(s, mountain));
        (
            can_cast(s, strike),
            has_action(
                s,
                |a| matches!(a, Action::PlayLand { object, .. } if *object == mountain),
            ),
        )
    };

    // No Wizard: hidden from both players, unseen in both views, locked.
    for viewer in [0, 1] {
        assert_eq!(shown_to(&s, viewer, strike), None);
        assert_eq!(shown_to(&s, viewer, mountain), None);
    }
    let library = s.game().state.zones.size(ZoneType::Library, PlayerId(0)) as u32;
    let owner = Observation::for_player(s.game(), PlayerId(0));
    let unseen: u32 = owner.agent.unseen_counts.as_ref().unwrap().values().sum();
    assert_eq!(unseen, library + 2);
    let constraints = HiddenHandConstraints::for_viewer(s.game(), PlayerId(1));
    assert_eq!((constraints.foretold, constraints.face_down_exile), (0, 2));
    assert_eq!(playable(&mut s), (false, false));

    // Determinizing for the caster resamples the hidden slots.
    let mut resampled = false;
    for seed in 0..32 {
        let mut game = s.game().clone();
        game.determinize(PlayerId(0), seed).unwrap();
        let exile = game
            .state
            .zones
            .zone_objects(ZoneType::Exile, PlayerId(0))
            .to_vec();
        assert_eq!(exile.len(), 3);
        let granted: Vec<_> = game
            .state
            .cast_permissions
            .grants
            .iter()
            .filter(|grant| grant.face_down)
            .map(|grant| grant.object)
            .collect();
        assert_eq!(granted.len(), 2);
        assert_ne!(granted[0], granted[1]);
        assert!(granted.iter().all(|object| exile.contains(object)));
        resampled |= !granted.contains(&strike) || !granted.contains(&mountain);
    }
    assert!(resampled);

    // Gandalf, a Wizard, shows them to his controller and lets them be played.
    cast_card(&mut s, gandalf);
    settle(&mut s);
    assert_eq!(shown_to(&s, 0, strike), Some((true, false)));
    assert_eq!(shown_to(&s, 0, mountain), Some((true, false)));
    assert_eq!(shown_to(&s, 1, strike), None);
    assert_eq!(playable(&mut s), (true, true));
    cast_card(&mut s, strike);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    s.assert_life(1, 16);
    let wizard = permanent_named(&s, "Gandalf, Goblins' Bane");
    let object = s.game().state.permanents[wizard].as_ref().unwrap().object;
    assert_eq!(s.game().characteristics(wizard).power, 3);

    // Gandalf leaves while another Wizard remains: still shown and playable.
    let mut other = s.clone();
    other.force_permanent_on_battlefield(0, "High Fae Trickster");
    move_card(&mut other, object, ZoneType::Graveyard);
    assert_eq!(shown_to(&other, 0, mountain), Some((true, false)));
    assert_eq!(shown_to(&other, 1, mountain), None);
    assert!(playable(&mut other).1);

    // The last Wizard leaves: hidden and locked; the next restores both.
    move_card(&mut s, object, ZoneType::Graveyard);
    assert_eq!(shown_to(&s, 0, mountain), None);
    assert_eq!(playable(&mut s).1, false);
    s.force_permanent_on_battlefield(0, "High Fae Trickster");
    assert_eq!(shown_to(&s, 0, mountain), Some((true, false)));
    assert!(playable(&mut s).1);
    choose(
        &mut s,
        |a| matches!(a, Action::PlayLand { object, .. } if *object == mountain),
    );
    assert_eq!(zone_of(&s, now(&s, mountain)), Some(ZoneType::Battlefield));
}

/// A foretold card and Flameshape's face-down cards share their owner's
/// exile. Determinizing for either player and materializing the opponent's
/// possible worlds refill both kinds of slot at once: every card stays in
/// one zone, the foretold slot holds a foretell card, and each grant names
/// its own exiled card.
#[test]
fn flameshape_and_foretold_slots_stay_disjoint_when_resampled() {
    use crate::support::{
        game::{move_from_library, pin_top},
        inspect::now,
        play::{cast_profile, choose},
    };
    use managym::{
        agent::action::Action,
        flow::game::Game,
        possible_worlds::PossibleWorldSpace,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Gandalf, Goblins' Bane", "Sozin's Comet"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let comet = move_from_library(&mut s, 0, "Sozin's Comet", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { object, .. } if *object == comet),
    );
    let gandalf = move_from_library(&mut s, 0, "Gandalf, Goblins' Bane", ZoneType::Hand);
    let top = pin_top(&mut s, &["Sozin's Comet", "Island"]);
    cast_profile(&mut s, gandalf, SpellProfile::Adventure);
    settle(&mut s);
    let original = [now(&s, comet), now(&s, top[0]), now(&s, top[1])];

    let check = |game: &Game| {
        let zones = [
            ZoneType::Library,
            ZoneType::Hand,
            ZoneType::Battlefield,
            ZoneType::Graveyard,
            ZoneType::Stack,
            ZoneType::Exile,
            ZoneType::Command,
        ];
        let mut seen = std::collections::BTreeMap::new();
        for player in [PlayerId(0), PlayerId(1)] {
            for zone in zones {
                for &object in game.state.zones.zone_objects(zone, player).iter() {
                    assert_eq!(game.state.zones.zone_of(object), Some(zone));
                    *seen.entry(object.slot()).or_insert(0) += 1;
                }
            }
        }
        assert!(seen.values().all(|count| *count == 1), "one zone per card");
        let exile = game
            .state
            .zones
            .zone_objects(ZoneType::Exile, PlayerId(0))
            .to_vec();
        assert_eq!(exile.len(), 4);
        let grants = &game.state.cast_permissions.grants;
        let hidden: Vec<_> = grants
            .iter()
            .filter(|grant| grant.foretold || grant.face_down)
            .collect();
        assert_eq!(hidden.len(), 3);
        let mut objects: Vec<_> = hidden.iter().map(|grant| grant.object).collect();
        objects.sort_unstable();
        objects.dedup();
        assert_eq!(objects.len(), 3, "each hidden grant names its own card");
        for grant in hidden {
            assert!(exile.contains(&grant.object));
            if grant.foretold {
                assert!(game.state[grant.object].foretell().is_some());
            }
        }
        exile
    };
    check(s.game());

    let mut moved = [false, false];
    for seed in 0..48 {
        for viewer in [0, 1] {
            let mut game = s.game().clone();
            game.determinize(PlayerId(viewer), seed).unwrap();
            let exile = check(&game);
            moved[viewer] |= original.iter().any(|object| !exile.contains(object));
        }
    }
    assert_eq!(moved, [true, true]);

    let space = PossibleWorldSpace::for_viewer(s.game(), PlayerId(1));
    assert_eq!(space.projection().face_down_exile, 2);
    for world in space.worlds().iter().take(8) {
        for seed in 0..4 {
            check(&space.materialize(s.game(), world, seed).unwrap());
        }
    }
}

/// A visible Flameshape card's permission reaches its controller's policy
/// input as a program naming its face-down and conditional terms, its JSON
/// round trip and the semantic text; the opponent sees neither the card nor
/// the program.
#[test]
fn flameshape_permission_terms_reach_the_policy_input() {
    use crate::support::{
        game::{move_from_library, pin_top},
        inspect::now,
        play::{cast_card, cast_profile},
    };
    use managym::{
        agent::{observation_encoder::PLAY_GRANT, observation_layout::fact},
        cards::tokens::name_value,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Gandalf, Goblins' Bane", "Lightning Strike"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let gandalf = move_from_library(&mut s, 0, "Gandalf, Goblins' Bane", ZoneType::Hand);
    let top = pin_top(&mut s, &["Lightning Strike", "Mountain"]);
    cast_profile(&mut s, gandalf, SpellProfile::Adventure);
    settle(&mut s);
    cast_card(&mut s, gandalf);
    settle(&mut s);
    let strike = now(&s, top[0]);
    let id = s.game().state[strike].id.0 as i32;

    let grant_details = |viewer: usize| {
        let observation = Observation::for_player(s.game(), PlayerId(viewer));
        let input = InputRows::read(&observation, &ObservationEncoderConfig::default()).unwrap();
        input
            .rows("facts")
            .into_iter()
            .filter(|row| {
                let id = ((row[fact::ID_HIGH] as u32) << 16) | row[fact::ID_LOW] as u32;
                id == name_value(PLAY_GRANT) as u32
            })
            .map(|row| {
                (
                    row[fact::OBJECT],
                    ((row[fact::DETAIL_HIGH] as u32) << 16) | row[fact::DETAIL_LOW] as u32,
                )
            })
            .collect::<Vec<_>>()
    };
    let owner = Observation::for_player(s.game(), PlayerId(0));
    let card = owner.agent_cards.iter().find(|card| card.id == id).unwrap();
    let permission = &card.cast_permissions[0];
    assert!(card.face_down && permission.face_down && !permission.foretold);
    assert!(permission.play_lands, "Flameshape says \"you may play\"");
    let condition = permission.condition.clone().expect("conditional grant");
    assert!(condition.contains("Wizard"), "{condition}");
    let restored: Observation = serde_json::from_str(&owner.to_json()).unwrap();
    assert_eq!(restored.to_json(), owner.to_json());

    // Only the Flameshape cards carry face-down programs here.
    let face_down = name_value("face_down") as u32;
    let details: Vec<u32> = grant_details(0)
        .into_iter()
        .map(|(_, detail)| detail)
        .collect();
    for name in ["face_down", "condition", "play_lands", condition.as_str()] {
        assert!(
            details.contains(&(name_value(name) as u32)),
            "{name} in the grant program"
        );
    }
    assert!(grant_details(1)
        .iter()
        .all(|(_, detail)| *detail != face_down));
}

/// Flameshape says "you may play": a land it exiles is played, the land back
/// of a modal card included, while the sorcery front is cast.
#[test]
fn flameshape_plays_a_land_and_a_modal_land_face() {
    use crate::support::{
        game::{move_from_library, pin_top},
        inspect::now,
        play::{cast_card, cast_profile},
    };
    use managym::{
        agent::action::Action,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Gandalf, Goblins' Bane", "Tidal Errand"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let gandalf = move_from_library(&mut s, 0, "Gandalf, Goblins' Bane", ZoneType::Hand);
    let top = pin_top(&mut s, &["Tidal Errand", "Mountain"]);
    cast_profile(&mut s, gandalf, SpellProfile::Adventure);
    settle(&mut s);
    cast_card(&mut s, gandalf);
    settle(&mut s);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    let (errand, mountain) = (now(&s, top[0]), now(&s, top[1]));
    let mut plays: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match *action {
            Action::PlayLand {
                object, profile, ..
            } if [errand, mountain].contains(&object) => Some(("play", object, profile)),
            Action::CastSpell {
                object, profile, ..
            } if [errand, mountain].contains(&object) => Some(("cast", object, profile)),
            _ => None,
        })
        .collect();
    plays.sort_by_key(|(verb, object, _)| (*verb, *object));
    let mut expected = vec![
        ("cast", errand, SpellProfile::Front),
        ("play", errand, SpellProfile::BackFace),
        ("play", mountain, SpellProfile::Front),
    ];
    expected.sort_by_key(|(verb, object, _)| (*verb, *object));
    assert_eq!(plays, expected);
}

/// A grant's price names the card's spell face: Kelp Hollow's front is a
/// land, so Flameshape shows Kelp Warden's {2}{U} and an airbend grant its
/// own {2}, never "no cost".
#[test]
fn a_grant_on_a_land_front_prices_its_spell_back() {
    use crate::support::{
        game::{move_from_library, pin_top},
        inspect::now,
        play::{cast_card, cast_profile},
    };
    use managym::{
        agent::observation::Observation,
        flow::cast_permission::PlayGrant,
        state::{mana::ManaCost, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Gandalf, Goblins' Bane", "Kelp Hollow"]);
    let gandalf = move_from_library(&mut s, 0, "Gandalf, Goblins' Bane", ZoneType::Hand);
    let top = pin_top(&mut s, &["Kelp Hollow"]);
    cast_profile(&mut s, gandalf, SpellProfile::Adventure);
    settle(&mut s);
    cast_card(&mut s, gandalf);
    settle(&mut s);
    let shaped = now(&s, top[0]);
    let airbent = move_from_library(&mut s, 0, "Kelp Hollow", ZoneType::Exile);
    s.game_mut().state.cast_permissions.grant(PlayGrant {
        cost: Some(ManaCost::parse("{2}")),
        ..PlayGrant::new(airbent, PlayerId(0))
    });
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let cost_of = |object: managym::state::game_object::ObjectId| {
        let id = s.game().state[object].id.0 as i32;
        let card = observation
            .agent_cards
            .iter()
            .find(|card| card.id == id)
            .unwrap();
        let [grant] = card.cast_permissions.as_slice() else {
            panic!("one grant on {object:?}");
        };
        grant.cost.clone()
    };
    assert_eq!(cost_of(shaped), Some(ManaCost::parse("{2}{U}")));
    assert_eq!(cost_of(airbent), Some(ManaCost::parse("{2}")));
}
