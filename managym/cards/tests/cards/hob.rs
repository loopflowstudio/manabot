//! Beorn's real combat trigger composes with Kellan's independently resolved upgrades.
use crate::support::{
    game::five_color_setup,
    inspect::permanent_named,
    play::{activate, enter_and_settle, pass_until, settle},
};
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
        let config = ObservationEncoderConfig {
            policy_input_version: 8,
            ..Default::default()
        };
        assert_eq!(
            encode(&obs, &config).unwrap(),
            encode(&Observation::for_player(&fork, PlayerId(0)), &config).unwrap()
        );
        for version in 1..8 {
            assert!(encode(
                &obs,
                &ObservationEncoderConfig {
                    policy_input_version: version,
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("select v8"));
        }
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
