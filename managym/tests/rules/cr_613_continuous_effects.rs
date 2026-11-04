//! CR 611 / 613: continuous effects from resolved spells and abilities are
//! records on `GameState::effects`. Layer 6 applies them by timestamp
//! (CR 613.7), each duration ends at its boundary (CR 514.2, CR 400.7), and
//! the records are hashed and forked like the rest of the match state.

use std::sync::Arc;

use managym::{
    agent::action::ActionSpaceKind,
    flow::turn::StepKind,
    state::{
        card::{Keyword, Keywords},
        effect::{Affected, EffectDuration, Modification},
        game_object::{PermanentId, PlayerId},
        zone::ZoneType,
    },
};

use crate::support::game::*;
use crate::support::inspect::card_of;

/// Player 0's Gray Ogre (no printed keywords) in player 0's main phase.
fn ogre_scenario() -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        613_007,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    (s, ogre)
}

fn flying() -> Modification {
    Modification::AddKeywords(Keywords::of(&[Keyword::Flying]))
}

fn indefinitely(s: &mut Scenario, id: PermanentId, modification: Modification) {
    let affected = s.game().affected_permanent(id).unwrap();
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected],
        modification,
        EffectDuration::Indefinite,
    );
}

#[test]
fn empty_effect_batch_keeps_cached_characteristics_but_allocates_a_timestamp() {
    let (mut s, ogre) = ogre_scenario();
    let before = s.game().characteristics(ogre);
    let hash = s.game().state.deterministic_hash();
    s.game_mut()
        .apply_continuous(None, PlayerId(0), [], flying(), EffectDuration::EndOfTurn);
    assert!(s.game().state.effects.is_empty());
    assert!(Arc::ptr_eq(&before, &s.game().characteristics(ogre)));
    assert_ne!(hash, s.game().state.deterministic_hash());
}

/// CR 613.7: a loss of all abilities removes grants with earlier timestamps
/// only; a later grant applies.
#[test]
fn cr_613_7_a_grant_later_than_ability_loss_applies_and_an_earlier_one_does_not() {
    let (mut later_grant, ogre) = ogre_scenario();
    until_end_of_turn(&mut later_grant, ogre, Modification::LoseAllAbilities);
    until_end_of_turn(&mut later_grant, ogre, flying());
    assert!(later_grant.game().characteristics(ogre).abilities_lost);
    assert!(later_grant
        .game()
        .characteristics(ogre)
        .keywords
        .has(Keyword::Flying));

    let (mut earlier_grant, ogre) = ogre_scenario();
    until_end_of_turn(&mut earlier_grant, ogre, flying());
    until_end_of_turn(&mut earlier_grant, ogre, Modification::LoseAllAbilities);
    assert!(earlier_grant.game().characteristics(ogre).abilities_lost);
    assert!(!earlier_grant
        .game()
        .characteristics(ogre)
        .keywords
        .has(Keyword::Flying));
}

/// CR 514.2: "until end of turn" records end in the cleanup step; the
/// permanent's characteristics return to its printed ones.
#[test]
fn cr_514_2_an_end_of_turn_record_ends_at_cleanup() {
    let (mut s, ogre) = ogre_scenario();
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: 2,
            toughness: 1,
        },
    );
    indefinitely(&mut s, ogre, flying());
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 3));
    assert_eq!(s.game().effects_on(ogre).count(), 2);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
    let remaining: Vec<_> = s
        .game()
        .effects_on(ogre)
        .map(|effect| (effect.modification.clone(), effect.duration))
        .collect();
    assert_eq!(
        remaining,
        [(flying(), EffectDuration::Indefinite)],
        "an indefinite record outlives cleanup"
    );
}

/// CR 400.7: an indefinite record on an object ends when it changes zones; the
/// card returns as a new object with no records.
#[test]
fn cr_400_7_an_indefinite_record_ends_when_its_object_changes_zones() {
    let (mut s, ogre) = ogre_scenario();
    indefinitely(&mut s, ogre, flying());
    let old = s.game().affected_permanent(ogre).unwrap();
    assert!(s.game().characteristics(ogre).keywords.has(Keyword::Flying));

    let card = card_of(&s, ogre);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let returned = s.game().state.card_to_permanent[card].expect("returned");
    assert_ne!(s.game().affected_permanent(returned), Some(old));
    assert!(s.game().state.effects.is_empty());
    assert_eq!(s.game().effects_on(returned).count(), 0);
    assert!(!s
        .game()
        .characteristics(returned)
        .keywords
        .has(Keyword::Flying));
}

/// Records are match state: restoring a fork snapshot removes a record
/// applied after it and restores records a zone change ended after it.
#[test]
fn fork_restore_restores_continuous_effect_records_and_the_state_hash() {
    let (s, ogre) = ogre_scenario();
    let mut branch = s.game().clone();
    let effects = branch.state.effects.clone();
    let hash = branch.state.deterministic_hash();

    let mark = branch.clone();
    let affected = branch.affected_permanent(ogre).unwrap();
    branch.apply_continuous(
        None,
        PlayerId(0),
        [affected],
        flying(),
        EffectDuration::EndOfTurn,
    );
    assert_ne!(branch.state.effects, effects);
    assert_ne!(branch.state.deterministic_hash(), hash);
    branch = mark;
    assert_eq!(branch.state.effects, effects);
    assert_eq!(branch.state.deterministic_hash(), hash);
    assert!(!branch.characteristics(ogre).keywords.has(Keyword::Flying));

    branch.apply_continuous(
        None,
        PlayerId(0),
        [affected],
        flying(),
        EffectDuration::Indefinite,
    );
    let effects = branch.state.effects.clone();
    let hash = branch.state.deterministic_hash();
    let mark = branch.clone();
    let card = branch.state.permanents[ogre].as_ref().unwrap().card;
    branch.move_card(card, ZoneType::Hand);
    assert!(branch.state.effects.is_empty());
    branch = mark;
    assert_eq!(branch.state.effects, effects);
    assert_eq!(branch.state.deterministic_hash(), hash);
    assert!(branch.characteristics(ogre).keywords.has(Keyword::Flying));
}

/// Equal games with equal records hash equally; the records are part of the
/// hash.
#[test]
fn equal_records_give_equal_deterministic_hashes() {
    let build = |records: bool| {
        let (mut s, ogre) = ogre_scenario();
        if records {
            until_end_of_turn(&mut s, ogre, Modification::LoseAllAbilities);
            until_end_of_turn(&mut s, ogre, flying());
            player_effect(
                &mut s,
                1,
                Modification::NoMaximumHandSize,
                EffectDuration::Indefinite,
            );
        }
        s
    };
    let (first, second) = (build(true), build(true));
    assert_eq!(first.game().state.effects, second.game().state.effects);
    assert_eq!(
        first.game().state.deterministic_hash(),
        second.game().state.deterministic_hash()
    );
    assert_ne!(
        first.game().state.deterministic_hash(),
        build(false).game().state.deterministic_hash()
    );
    assert!(first
        .game()
        .state
        .effects
        .iter()
        .any(|effect| effect.affected == Affected::Player(PlayerId(1))));
}

/// CR 205.4, 704.5j: a permanent that isn't legendary through an effect is
/// outside the legend rule while the effect lasts.
#[test]
fn a_nonlegendary_record_exempts_its_permanent_from_the_legend_rule() {
    let mut s = ur_lessons_setup();
    let first = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    until_end_of_turn(&mut s, first, Modification::Nonlegendary);
    let second = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    s.pass_priority();
    assert_ne!(s.action_space().kind, ActionSpaceKind::LegendRule);
    for id in [first, second] {
        assert_eq!(
            s.game().state.zones.zone_of(card_of(&s, id)),
            Some(ZoneType::Battlefield)
        );
    }
}

/// Shared payloads are admitted before the old per-keyword/per-object gate.
#[test]
fn shared_effects_admit_140_and_270_fragments_without_changing_rules_records() {
    use managym::agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    };
    for (activations, targets, version, capacity) in [(7, 10, 5, 128), (9, 15, 6, 256)] {
        let (mut s, _) = ogre_scenario();
        let objects: Vec<_> = (0..targets)
            .map(|_| s.force_permanent_on_battlefield(1, "Mountain"))
            .collect();
        let affected: Vec<_> = objects
            .iter()
            .map(|id| s.game().affected_permanent(*id).unwrap())
            .collect();
        for _ in 0..activations {
            s.game_mut().apply_continuous(
                None,
                PlayerId(0),
                affected.iter().copied(),
                Modification::LoseKeywords(Keywords::of(&[
                    Keyword::Hexproof,
                    Keyword::Indestructible,
                ])),
                EffectDuration::EndOfTurn,
            );
        }
        let before = s.game().state.deterministic_hash();
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            let config = ObservationEncoderConfig {
                policy_input_version: version,
                max_facts: capacity,
                ..Default::default()
            };
            assert!(encode(&obs, &config)
                .unwrap_err()
                .to_string()
                .contains("facts"));
            let shared = obs.shared_effects.as_ref().unwrap();
            assert_eq!(shared.len(), activations);
            assert!(
                shared
                    .iter()
                    .all(|effect| effect.effect.keywords.len() == 2
                        && effect.members.len() == targets)
            );
            let encoded = encode(
                &obs,
                &ObservationEncoderConfig {
                    policy_input_version: 7,
                    ..config
                },
            )
            .unwrap();
            assert_eq!(
                encoded
                    .effects
                    .as_chunks::<18>()
                    .0
                    .iter()
                    .filter(|row| row[17] == 1.0)
                    .count(),
                activations
            );
            assert_eq!(
                encoded
                    .effect_members
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .filter(|row| row[0] > 0.0)
                    .count(),
                activations * targets
            );
            let mut orders: Vec<_> = shared
                .iter()
                .flat_map(|record| record.members.iter().map(|member| member.order))
                .collect();
            orders.sort_unstable();
            assert_eq!(orders, (0..activations * targets).collect::<Vec<_>>());
        }
        assert_eq!(s.game().state.effects.iter().count(), activations * targets);
        assert_eq!(s.game().state.deterministic_hash(), before);
    }
}

#[test]
fn shared_effects_preserve_duplicate_additive_memberships_and_later_entrants() {
    use managym::agent::shared_effects::project;
    let (mut s, ogre) = ogre_scenario();
    let affected = s.game().affected_permanent(ogre).unwrap();
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected, affected],
        Modification::ModifyPt {
            power: 1,
            toughness: 1,
        },
        EffectDuration::EndOfTurn,
    );
    let later = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let shared = project(s.game());
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].members.len(), 2);
    assert_eq!(shared[0].members[0].target, shared[0].members[1].target);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    assert_eq!(s.game().characteristics(later).pt(), (2, 2));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(project(s.game()).is_empty());
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
}

#[test]
fn shared_effects_keep_intervening_grants_and_different_expiries() {
    use managym::agent::shared_effects::project;
    let (mut s, ogre) = ogre_scenario();
    indefinitely(&mut s, ogre, Modification::LoseAllAbilities);
    indefinitely(&mut s, ogre, flying());
    until_end_of_turn(&mut s, ogre, Modification::LoseAllAbilities);
    let shared = project(s.game());
    assert_eq!(shared.len(), 3);
    assert_eq!(shared[0].effect.modification, shared[2].effect.modification);
    assert_ne!(shared[0].effect.duration, shared[2].effect.duration);
    assert!(shared
        .windows(2)
        .all(|pair| pair[0].effect.timestamp < pair[1].effect.timestamp));
    assert!(!s.game().characteristics(ogre).keywords.has(Keyword::Flying));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(project(s.game()).len(), 2);
    assert!(s.game().characteristics(ogre).keywords.has(Keyword::Flying));
}

/// Equal old payloads bound to different exact sources have different futures.
#[test]
fn shared_effects_bind_exact_sources_before_and_after_a_zone_change() {
    use managym::agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    };
    use managym::state::effect::ContinuousEffect;
    let (mut s, target) = ogre_scenario();
    let a = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let b = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let source_card = card_of(&s, a);
    let affected = s.game().affected_permanent(target).unwrap();
    let mut games = [s.game().clone(), s.game().clone()];
    for (game, source) in games.iter_mut().zip([a, b]) {
        let timestamp = game.state.id_gen.next_id();
        let record = ContinuousEffect {
            timestamp,
            source: Some(game.state.cards[source_card].definition_id),
            controller: PlayerId(0),
            affected,
            modification: flying(),
            duration: EffectDuration::WhileYouControlSource,
            source_object: game.permanent_object_ref(source),
        };
        game.state.effects.push(record);
        assert!(game.characteristics(target).keywords.has(Keyword::Flying));
    }
    for viewer in [PlayerId(0), PlayerId(1)] {
        let observations = games
            .each_ref()
            .map(|game| Observation::for_player(game, viewer));
        let old = ObservationEncoderConfig {
            policy_input_version: 6,
            ..Default::default()
        };
        assert_eq!(
            encode(&observations[0], &old).unwrap(),
            encode(&observations[1], &old).unwrap()
        );
        let new = ObservationEncoderConfig {
            policy_input_version: 7,
            ..old
        };
        let encoded = observations
            .each_ref()
            .map(|obs| encode(obs, &new).unwrap());
        assert_ne!(encoded[0].effects[12], encoded[1].effects[12]);
    }
    for game in &mut games {
        game.move_card(source_card, ZoneType::Hand);
    }
    assert!(!games[0]
        .characteristics(target)
        .keywords
        .has(Keyword::Flying));
    assert!(games[1]
        .characteristics(target)
        .keywords
        .has(Keyword::Flying));
}
