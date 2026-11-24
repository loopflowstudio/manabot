//! Beorn's real combat trigger composes with Kellan's independently resolved upgrades.
use crate::support::{
    game::five_color_setup,
    inspect::permanent_named,
    play::{activate, enter_and_settle, pass_until, settle},
};
use managym::state::card::SpellProfile;
use managym::{
    agent::{
        action::ActionSpaceKind,
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
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
            encode(&obs, &config).unwrap(),
            encode(&Observation::for_player(&fork, PlayerId(0)), &config).unwrap()
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
