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

/// Install an isolated rules witness without registering historical cards in
/// the playable world. Its permanent and exact object identity stay real.
fn layer_fixture(
    s: &mut Scenario,
    name: &str,
    definition: managym::state::card::CardDefinition,
) -> PermanentId {
    use managym::state::card::Card;
    let id = s.force_permanent_on_battlefield(0, name);
    let card_id = card_of(s, id);
    let old = &s.game().state.cards[card_id];
    let card = Card::from_definition(old.id, old.owner, old.definition_id, Arc::new(definition));
    s.game_mut().state.cards[card_id] = card;
    id
}

fn continuous(
    scope: managym::state::card::StaticScope,
    modifications: Vec<Modification<managym::state::card::LayerValue>>,
) -> managym::state::card::StaticAbility {
    use managym::state::card::{StaticAbility, StaticContinuous};
    StaticAbility::Continuous(StaticContinuous {
        scope,
        condition: None,
        characteristic_defining: false,
        modifications,
    })
}

#[test]
fn cr_613_dependency_overrides_timestamps_and_continues_across_layers() {
    use managym::state::{
        ability::Condition,
        card::{
            CardDefinition, CardType, CardTypes, LayerValue, StaticAbility, StaticLandType,
            StaticScope,
        },
        predicate::CardPredicate,
    };
    for moon_first in [false, true] {
        let (mut s, _) = ogre_scenario();
        let moon = CardDefinition {
            name: "Moon witness".into(),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::SetsLandType(StaticLandType {
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    nonbasic: true,
                    ..Default::default()
                },
                subtype: "Mountain".into(),
                condition: Condition::Always,
            })],
            ..Default::default()
        };
        let urborg = CardDefinition {
            name: "Urborg witness".into(),
            types: CardTypes::new([CardType::Land]),
            statics: vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Modification::AddSubtypes(vec!["Swamp".into()])],
            )],
            ..Default::default()
        };
        let (moon_id, urborg_id) = if moon_first {
            let m = layer_fixture(&mut s, "Gray Ogre", moon);
            let u = layer_fixture(&mut s, "Mountain", urborg);
            (m, u)
        } else {
            let u = layer_fixture(&mut s, "Mountain", urborg);
            let m = layer_fixture(&mut s, "Gray Ogre", moon);
            (m, u)
        };
        assert!(s.game().characteristics(urborg_id).has_subtype("Mountain"));
        assert!(!s.game().characteristics(urborg_id).has_subtype("Swamp"));
        // Ability removal in layer 6 cannot erase the layer-4 Moon effect.
        indefinitely(&mut s, moon_id, Modification::LoseAllAbilities);
        assert!(s.game().characteristics(urborg_id).has_subtype("Mountain"));
        let card = card_of(&s, moon_id);
        s.game_mut().move_card(card, ZoneType::Graveyard);
        assert!(s.game().characteristics(urborg_id).has_subtype("Swamp"));
    }
    // Lattice/March: March depends on Lattice, even with the older timestamp.
    for lattice_first in [false, true] {
        let (mut s, _) = ogre_scenario();
        let lattice = CardDefinition {
            name: "Lattice witness".into(),
            types: CardTypes::new([CardType::Artifact]),
            mana_cost: Some(managym::state::mana::ManaCost::parse("{6}")),
            statics: vec![continuous(
                StaticScope::All(CardPredicate::default()),
                vec![Modification::AddTypes(vec![CardType::Artifact])],
            )],
            ..Default::default()
        };
        let march = CardDefinition {
            name: "March witness".into(),
            types: CardTypes::new([CardType::Enchantment]),
            mana_cost: Some(managym::state::mana::ManaCost::parse("{3}{U}")),
            statics: vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Artifact),
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                }),
                vec![
                    Modification::BecomeCreature,
                    Modification::SetBasePt {
                        power: Some(LayerValue::ManaValue),
                        toughness: Some(LayerValue::ManaValue),
                    },
                ],
            )],
            ..Default::default()
        };
        let (l, m) = if lattice_first {
            let l = layer_fixture(&mut s, "Gray Ogre", lattice);
            let m = layer_fixture(&mut s, "Gray Ogre", march);
            (l, m)
        } else {
            let m = layer_fixture(&mut s, "Gray Ogre", march);
            let l = layer_fixture(&mut s, "Gray Ogre", lattice);
            (l, m)
        };
        assert_eq!(s.game().characteristics(l).pt(), (6, 6));
        assert_eq!(s.game().characteristics(m).pt(), (4, 4));
        let land = s.force_permanent_on_battlefield(0, "Mountain");
        assert!(s.game().characteristics(land).is_creature());
        assert_eq!(s.game().characteristics(land).pt(), (0, 0));
        s.game_mut().scenario_refresh_priority().unwrap();
        s.pass_priority();
        assert!(s.game().state.permanents[land].is_none());
    }
}

fn witness(
    name: &str,
    types: &[managym::state::card::CardType],
    cost: &str,
    statics: Vec<managym::state::card::StaticAbility>,
) -> managym::state::card::CardDefinition {
    managym::state::card::CardDefinition {
        name: name.into(),
        types: managym::state::card::CardTypes::new(types.iter().copied()),
        mana_cost: Some(managym::state::mana::ManaCost::parse(cost)),
        statics,
        ..Default::default()
    }
}

#[test]
fn cr_613_humility_opalescence_and_dress_down_cda_timestamp_orders() {
    use managym::state::card::{CardType, LayerValue, StaticScope};
    use managym::state::predicate::CardPredicate;
    for humility_first in [false, true] {
        let (mut s, _) = ogre_scenario();
        let humility = witness(
            "Humility",
            &[CardType::Enchantment],
            "{2}{W}{W}",
            vec![continuous(
                StaticScope::All(CardPredicate::creature()),
                vec![
                    Modification::LoseAllAbilities,
                    Modification::SetBasePt {
                        power: Some(LayerValue::Fixed(1)),
                        toughness: Some(LayerValue::Fixed(1)),
                    },
                ],
            )],
        );
        let opalescence = witness(
            "Opalescence",
            &[CardType::Enchantment],
            "{2}{W}{W}",
            vec![continuous(
                StaticScope::Other(CardPredicate {
                    card_type: Some(CardType::Enchantment),
                    ..Default::default()
                }),
                vec![
                    Modification::BecomeCreature,
                    Modification::SetBasePt {
                        power: Some(LayerValue::ManaValue),
                        toughness: Some(LayerValue::ManaValue),
                    },
                ],
            )],
        );
        let (h, o) = if humility_first {
            let h = layer_fixture(&mut s, "Gray Ogre", humility);
            let o = layer_fixture(&mut s, "Gray Ogre", opalescence);
            (h, o)
        } else {
            let o = layer_fixture(&mut s, "Gray Ogre", opalescence);
            let h = layer_fixture(&mut s, "Gray Ogre", humility);
            (h, o)
        };
        let h_view = s.game().characteristics(h);
        assert!(h_view.abilities_lost);
        assert_eq!(h_view.pt(), if humility_first { (4, 4) } else { (1, 1) });
        assert!(!s.game().characteristics(o).is_creature());
    }
    let (mut s, _) = ogre_scenario();
    let mut goyf = witness(
        "Tarmogoyf",
        &[CardType::Creature],
        "{1}{G}",
        vec![managym::state::card::StaticAbility::Continuous(
            managym::state::card::StaticContinuous {
                scope: StaticScope::This,
                condition: None,
                characteristic_defining: true,
                modifications: vec![Modification::SetBasePt {
                    power: Some(LayerValue::GraveyardCardTypes { offset: 0 }),
                    toughness: Some(LayerValue::GraveyardCardTypes { offset: 1 }),
                }],
            },
        )],
    );
    goyf.toughness = Some(1);
    let g = layer_fixture(&mut s, "Gray Ogre", goyf);
    let card = crate::support::inspect::library_card(&s, 0, "Mountain");
    s.game_mut().move_card(card, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(g).pt(), (1, 2));
    let dress = layer_fixture(
        &mut s,
        "Gray Ogre",
        witness(
            "Dress Down",
            &[CardType::Enchantment],
            "{1}{U}",
            vec![continuous(
                StaticScope::All(CardPredicate::creature()),
                vec![Modification::LoseAllAbilities],
            )],
        ),
    );
    assert_eq!(s.game().characteristics(g).pt(), (0, 1));
    let card = card_of(&s, dress);
    s.game_mut().move_card(card, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(g).pt(), (2, 3));
}

#[test]
fn cr_613_ygra_imprisoned_retains_food_grants_and_painter_retains_color() {
    use managym::state::{
        ability::{Effect, PaymentCost},
        card::{ActivatedAbilityDefinition, CardType, StaticAbility, StaticScope},
        mana::{Color, ManaCost},
        predicate::CardPredicate,
    };
    for aura_first in [false, true] {
        let (mut s, ogre) = ogre_scenario();
        let food = ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{2}"),
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::gain_life(3)],
            ..Default::default()
        };
        let ygra = witness(
            "Ygra",
            &[CardType::Creature],
            "{3}{B}{G}",
            vec![
                StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}"))),
                continuous(
                    StaticScope::Other(CardPredicate::creature()),
                    vec![
                        Modification::AddTypes(vec![CardType::Artifact]),
                        Modification::AddSubtypes(vec!["Food".into()]),
                        Modification::GrantActivated(Box::new(food)),
                    ],
                ),
            ],
        );
        let mut aura = witness(
            "Imprisoned",
            &[CardType::Enchantment],
            "{2}{U}",
            vec![continuous(
                StaticScope::Attached,
                vec![
                    Modification::SetTypes {
                        types: vec![CardType::Land],
                        subtypes: vec![],
                    },
                    Modification::LoseAllAbilities,
                    Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                        tap_source: true,
                        effects: vec![Effect::add_mana(managym::state::mana::Mana::parse("C"))],
                        ..Default::default()
                    })),
                ],
            )],
        );
        aura.subtypes = vec!["Aura".into()];
        let (y, a) = if aura_first {
            let a = layer_fixture(&mut s, "Gray Ogre", aura);
            let y = layer_fixture(&mut s, "Gray Ogre", ygra);
            (y, a)
        } else {
            let y = layer_fixture(&mut s, "Gray Ogre", ygra);
            let a = layer_fixture(&mut s, "Gray Ogre", aura);
            (y, a)
        };
        s.game_mut().state.permanents[a]
            .as_mut()
            .unwrap()
            .attached_to = s.game().permanent_object_ref(y);
        let view = s.game().characteristics(y);
        assert!(view.is_land() && !view.is_creature() && view.ward.is_none());
        assert_eq!(view.activated_abilities.len(), 1);
        let other = s.game().characteristics(ogre);
        assert!(other.has_subtype("Food") && other.types.is_artifact());
        assert_eq!(other.activated_abilities.len(), 1);
        assert!(other.activated_abilities[0].sacrifice_source);
    }
    let (mut s, ogre) = ogre_scenario();
    let mut painter = witness(
        "Painter",
        &[CardType::Creature],
        "{2}",
        vec![continuous(
            StaticScope::All(CardPredicate::default()),
            vec![Modification::AddColors([Color::Blue].into_iter().collect())],
        )],
    );
    painter.power = Some(1);
    painter.toughness = Some(3);
    let p = layer_fixture(&mut s, "Gray Ogre", painter);
    indefinitely(&mut s, p, Modification::LoseAllAbilities);
    assert!(s.game().characteristics(ogre).colors.contains(&Color::Blue));
    let card = card_of(&s, p);
    s.game_mut().move_card(card, ZoneType::Graveyard);
    assert!(!s.game().characteristics(ogre).colors.contains(&Color::Blue));
    // Modifiers and counters precede the layer-7d switch.
    indefinitely(
        &mut s,
        ogre,
        Modification::SetBasePt {
            power: Some(1),
            toughness: Some(4),
        },
    );
    indefinitely(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: 2,
            toughness: 0,
        },
    );
    indefinitely(&mut s, ogre, Modification::SwitchPt);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 3));
}

#[test]
fn cr_613_sun_urborg_magus_and_saga_keep_earlier_layers_and_external_grants() {
    use managym::state::{
        ability::{Ability, Condition, Effect, TriggerCondition},
        card::{ActivatedAbilityDefinition, CardType, StaticAbility, StaticLandType, StaticScope},
        mana::Mana,
        predicate::CardPredicate,
    };
    for sun_first in [false, true] {
        let (mut s, _) = ogre_scenario();
        let urborg = witness(
            "Urborg",
            &[CardType::Land],
            "{0}",
            vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Modification::AddSubtypes(vec!["Swamp".into()])],
            )],
        );
        let sun = witness(
            "Sun",
            &[CardType::Enchantment],
            "{2}{R}",
            vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Modification::LoseNonManaAbilities],
            )],
        );
        let u = if sun_first {
            layer_fixture(&mut s, "Gray Ogre", sun);
            layer_fixture(&mut s, "Mountain", urborg)
        } else {
            let u = layer_fixture(&mut s, "Mountain", urborg);
            layer_fixture(&mut s, "Gray Ogre", sun);
            u
        };
        assert!(s.game().characteristics(u).has_subtype("Swamp"));
        assert_eq!(s.game().characteristics(u).activated_abilities.len(), 1);
    }
    let (mut s, _) = ogre_scenario();
    let mut moon = witness(
        "Magus",
        &[CardType::Creature],
        "{2}{R}",
        vec![StaticAbility::SetsLandType(StaticLandType {
            predicate: CardPredicate {
                card_type: Some(CardType::Land),
                nonbasic: true,
                ..Default::default()
            },
            subtype: "Mountain".into(),
            condition: Condition::Always,
        })],
    );
    moon.power = Some(2);
    moon.toughness = Some(2);
    let m = layer_fixture(&mut s, "Gray Ogre", moon);
    let mut saga = witness(
        "Saga",
        &[CardType::Land, CardType::Enchantment],
        "{0}",
        vec![],
    );
    saga.subtypes = vec!["Saga".into()];
    saga.abilities = vec![Ability::Triggered {
        condition: TriggerCondition::Chapter { chapter: 3 },
        intervening_if: None,
        effects: vec![Effect::DrawCards {
            count: managym::state::ability::EffectValue::Fixed(1),
        }],
    }];
    let u = layer_fixture(&mut s, "Mountain", saga);
    indefinitely(&mut s, m, Modification::LoseAllAbilities);
    assert!(s.game().characteristics(u).has_subtype("Mountain"));
    assert!(s.game().characteristics(u).triggered_abilities.is_empty());
    indefinitely(
        &mut s,
        u,
        Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::add_mana(Mana::parse("C"))],
            ..Default::default()
        })),
    );
    indefinitely(
        &mut s,
        u,
        Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
            mana_cost: managym::state::mana::ManaCost::parse("{2}"),
            tap_source: true,
            // Existing Construct token witnesses the retained creation program;
            // Saga-specific token sizing is independent of this layer interaction.
            effects: vec![Effect::create_tokens("Ballistic Boulder", 1)],
            ..Default::default()
        })),
    );
    assert_eq!(s.game().characteristics(u).activated_abilities.len(), 3);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(
        s.game().state.permanents[u]
            .as_ref()
            .unwrap()
            .counter_count(managym::state::permanent::CounterKind::Lore),
        0
    );
}

#[test]
fn cr_613_dress_down_enters_animated_without_draw_or_end_step_trigger() {
    use managym::state::{
        ability::{Ability, Effect, EffectValue, TriggerCondition, TriggerSubject},
        card::{CardType, LayerValue, StaticScope},
        predicate::CardPredicate,
    };
    let (mut s, _) = ogre_scenario();
    layer_fixture(
        &mut s,
        "Gray Ogre",
        witness(
            "Opalescence",
            &[CardType::Enchantment],
            "{2}{W}{W}",
            vec![continuous(
                StaticScope::Other(CardPredicate {
                    card_type: Some(CardType::Enchantment),
                    ..Default::default()
                }),
                vec![
                    Modification::BecomeCreature,
                    Modification::SetBasePt {
                        power: Some(LayerValue::ManaValue),
                        toughness: Some(LayerValue::ManaValue),
                    },
                ],
            )],
        ),
    );
    let mut dress = witness(
        "Dress Down",
        &[CardType::Enchantment],
        "{1}{U}",
        vec![continuous(
            StaticScope::All(CardPredicate::creature()),
            vec![Modification::LoseAllAbilities],
        )],
    );
    dress.abilities = vec![
        Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        },
        Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourEndStep,
            intervening_if: None,
            effects: vec![Effect::sacrifice_this()],
        },
    ];
    let d = layer_fixture(&mut s, "Gray Ogre", dress);
    let card = card_of(&s, d);
    s.game_mut().move_card(card, ZoneType::Hand);
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    crate::support::play::settle(&mut s);
    let id = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(s.game().characteristics(id).pt(), (2, 2));
    assert!(s.game().characteristics(id).triggered_abilities.is_empty());
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand - 1
    );
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
}

#[test]
fn cr_613_tap_condition_invalidates_cache_and_dependency_loop_uses_timestamps() {
    use managym::state::{
        ability::{Condition, Effect},
        card::{
            ActivatedAbilityDefinition, CardType, LayerValue, StaticAbility, StaticContinuous,
            StaticScope,
        },
        mana::Mana,
        predicate::CardPredicate,
    };
    let (mut s, ogre) = ogre_scenario();
    let mut definition = witness(
        "Tap witness",
        &[CardType::Artifact],
        "{1}",
        vec![StaticAbility::Continuous(StaticContinuous {
            scope: StaticScope::All(CardPredicate::creature()),
            condition: Some(Condition::SourceTapped),
            characteristic_defining: false,
            modifications: vec![Modification::ModifyPt {
                power: LayerValue::Fixed(2),
                toughness: LayerValue::Fixed(2),
            }],
        })],
    );
    definition.activated_abilities = vec![ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![Effect::add_mana(Mana::parse("C"))],
        ..Default::default()
    }];
    let source = layer_fixture(&mut s, "Gray Ogre", definition);
    let before = s.game().characteristics(ogre);
    assert_eq!(before.pt(), (2, 2));
    crate::support::play::activate(&mut s, source, 0);
    crate::support::play::settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    assert_eq!(before.pt(), (2, 2));
    for blue_first in [false, true] {
        let (mut s, ogre) = ogre_scenario();
        let to_artifact = witness(
            "Artifact loop",
            &[CardType::Enchantment],
            "{1}",
            vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Enchantment),
                    ..Default::default()
                }),
                vec![Modification::AddTypes(vec![CardType::Artifact])],
            )],
        );
        let to_enchantment = witness(
            "Enchantment loop",
            &[CardType::Artifact],
            "{1}",
            vec![continuous(
                StaticScope::All(CardPredicate {
                    card_type: Some(CardType::Artifact),
                    ..Default::default()
                }),
                vec![Modification::AddTypes(vec![CardType::Enchantment])],
            )],
        );
        let (a, b) = if blue_first {
            let a = layer_fixture(&mut s, "Gray Ogre", to_artifact);
            let b = layer_fixture(&mut s, "Gray Ogre", to_enchantment);
            (a, b)
        } else {
            let b = layer_fixture(&mut s, "Gray Ogre", to_enchantment);
            let a = layer_fixture(&mut s, "Gray Ogre", to_artifact);
            (a, b)
        };
        for id in [a, b] {
            let view = s.game().characteristics(id);
            assert!(view.types.is_artifact() && view.types.is_enchantment());
        }
        assert!(!s.game().characteristics(ogre).types.is_artifact());
    }
}

#[test]
fn cr_613_dependency_loop_keeps_timestamp_order_and_copy_ignores_grants_and_loss() {
    use managym::state::{
        ability::Effect,
        card::{ActivatedAbilityDefinition, CardType, StaticScope},
        predicate::CardPredicate,
    };
    for reverse in [false, true] {
        let (mut s, _) = ogre_scenario();
        let make = |name| {
            let mut d = witness(
                name,
                &[CardType::Creature],
                "{2}",
                vec![continuous(
                    StaticScope::Other(CardPredicate::creature()),
                    vec![Modification::LoseAllAbilities],
                )],
            );
            d.power = Some(2);
            d.toughness = Some(2);
            d
        };
        let early = layer_fixture(&mut s, "Gray Ogre", make(if reverse { "B" } else { "A" }));
        let late = layer_fixture(&mut s, "Gray Ogre", make(if reverse { "A" } else { "B" }));
        assert!(!s.game().characteristics(early).abilities_lost);
        assert!(s.game().characteristics(late).abilities_lost);
    }
    let (mut s, source) = ogre_scenario();
    let card = card_of(&s, source);
    s.game_mut().state.cards[card].keywords = Keywords::of(&[Keyword::Flying]);
    indefinitely(&mut s, source, Modification::LoseAllAbilities);
    indefinitely(
        &mut s,
        source,
        Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
            effects: vec![Effect::CreateTokenCopy {
                of: None,
                exceptions: vec![],
            }],
            ..Default::default()
        })),
    );
    assert!(!s
        .game()
        .characteristics(source)
        .keywords
        .has(Keyword::Flying));
    crate::support::play::activate(&mut s, source, 0);
    crate::support::play::settle(&mut s);
    let copy = s
        .battlefield_permanents_named(0, "Gray Ogre")
        .into_iter()
        .find(|&id| s.game().state.cards[card_of(&s, id)].is_token)
        .unwrap();
    let view = s.game().characteristics(copy);
    assert!(view.keywords.has(Keyword::Flying));
    assert!(!view.abilities_lost && view.activated_abilities.is_empty());
}

#[test]
fn cr_613_moon_after_saga_chapters_preserves_their_resolved_abilities() {
    use managym::state::{
        ability::{Ability, Condition, Effect, ObjectSelector, TriggerCondition},
        card::{ActivatedAbilityDefinition, CardType, StaticAbility, StaticLandType},
        mana::{Mana, ManaCost},
        permanent::CounterKind,
        predicate::CardPredicate,
    };
    let (mut s, _) = ogre_scenario();
    let grants = [
        ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::add_mana(Mana::parse("C"))],
            ..Default::default()
        },
        ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{2}"),
            // Existing Construct token witnesses the retained creation program;
            // Saga-specific token sizing is independent of this layer interaction.
            effects: vec![Effect::create_tokens("Ballistic Boulder", 1)],
            ..Default::default()
        },
    ];
    let mut saga = witness(
        "Saga",
        &[CardType::Land, CardType::Enchantment],
        "{0}",
        vec![],
    );
    saga.subtypes = vec!["Saga".into()];
    saga.abilities = grants
        .into_iter()
        .enumerate()
        .map(|(i, grant)| Ability::Triggered {
            condition: TriggerCondition::Chapter {
                chapter: (i + 1) as u8,
            },
            intervening_if: None,
            effects: vec![Effect::indefinitely(
                ObjectSelector::This,
                Modification::GrantActivated(Box::new(grant)),
            )],
        })
        .collect();
    saga.abilities.push(Ability::Triggered {
        condition: TriggerCondition::Chapter { chapter: 3 },
        intervening_if: None,
        effects: vec![Effect::DrawCards {
            count: managym::state::ability::EffectValue::Fixed(1),
        }],
    });
    let old = layer_fixture(&mut s, "Mountain", saga);
    let card = card_of(&s, old);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    crate::support::play::settle(&mut s);
    let saga = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(
        s.game().state.permanents[saga]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        1
    );
    assert_eq!(s.game().characteristics(saga).activated_abilities.len(), 1);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.pass_priority();
    crate::support::play::settle(&mut s);
    assert_eq!(s.game().characteristics(saga).activated_abilities.len(), 2);
    let moon = witness(
        "Moon",
        &[CardType::Enchantment],
        "{2}{R}",
        vec![StaticAbility::SetsLandType(StaticLandType {
            predicate: CardPredicate {
                card_type: Some(CardType::Land),
                nonbasic: true,
                ..Default::default()
            },
            subtype: "Mountain".into(),
            condition: Condition::Always,
        })],
    );
    layer_fixture(&mut s, "Gray Ogre", moon);
    assert_eq!(s.game().characteristics(saga).activated_abilities.len(), 3);
    s.set_player_mana_pool(0, "2");
    // Mountain's intrinsic mana is index 0, then chapter I and chapter II.
    crate::support::play::activate(&mut s, saga, 2);
    crate::support::play::settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Ballistic Boulder").len(),
        1
    );
    assert!(s
        .game()
        .characteristics(saga)
        .triggered_abilities
        .is_empty());
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(
        s.game().state.permanents[saga]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        2
    );
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
}

#[test]
fn cr_613_granted_activation_retains_program_after_sacrificing_source_and_suspending() {
    use managym::{
        agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
        },
        state::{ability::Effect, card::ActivatedAbilityDefinition},
    };
    let (mut s, source) = ogre_scenario();
    indefinitely(
        &mut s,
        source,
        Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
            sacrifice_source: true,
            effects: vec![Effect::OptionalDiscardDraw { count: 1, up_to: 1 }],
            ..Default::default()
        })),
    );
    crate::support::play::activate(&mut s, source, 0);
    assert!(s.game().state.permanents[source].is_none());
    let config = ObservationEncoderConfig {
        policy_input_version: 8,
        ..Default::default()
    };
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(obs
        .continuation
        .iter()
        .any(|frame| frame.kind == 2 && !frame.program.is_empty()));
    assert!(obs.shared_effects.as_ref().unwrap().is_empty());
    assert!(encode(&obs, &ObservationEncoderConfig::default())
        .unwrap_err()
        .to_string()
        .contains("select v8"));
    assert!(encode(&obs, &config).is_ok());
    for _ in 0..4 {
        if s.game().state.suspended_decision.is_some() {
            break;
        }
        s.pass_priority();
    }
    assert!(s.game().state.suspended_decision.is_some());
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(obs
        .continuation
        .iter()
        .any(|frame| frame.kind == 5 && !frame.program.is_empty()));
    let fork = s.game().clone();
    assert_eq!(
        encode(&obs, &config).unwrap(),
        encode(&Observation::for_player(&fork, PlayerId(0)), &config).unwrap()
    );
    crate::support::play::settle(&mut s);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn cr_613_source_relative_alternatives_agree_in_layers_and_rule_restrictions() {
    use managym::state::{
        card::{CardType, LayerValue, Restriction, StaticAbility, StaticRestriction, StaticScope},
        predicate::CardPredicate,
    };
    let (mut s, ogre) = ogre_scenario();
    let scope = StaticScope::All(CardPredicate {
        any_of: vec![CardPredicate {
            namesake: true,
            ..Default::default()
        }],
        ..Default::default()
    });
    layer_fixture(
        &mut s,
        "Gray Ogre",
        witness(
            "Gray Ogre",
            &[CardType::Enchantment],
            "{2}",
            vec![
                continuous(
                    scope.clone(),
                    vec![Modification::ModifyPt {
                        power: LayerValue::Fixed(2),
                        toughness: LayerValue::Fixed(2),
                    }],
                ),
                StaticAbility::Restriction(StaticRestriction {
                    scope,
                    kind: Restriction::DoesntUntap,
                    unless: None,
                }),
            ],
        ),
    );
    // Both the layer computation and the later rule question must keep the
    // source name while descending into the alternative predicate.
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = true;
    let land = s.force_permanent_on_battlefield(0, "Mountain");
    s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.game().state.permanents[ogre].as_ref().unwrap().tapped);
    assert!(!s.game().state.permanents[land].as_ref().unwrap().tapped);
}

#[test]
fn cr_613_keyword_predicate_dependency_overrides_grant_timestamp() {
    use managym::state::{
        card::{CardType, StaticScope},
        predicate::CardPredicate,
    };
    for flying_first in [true, false] {
        let (mut s, ogre) = ogre_scenario();
        let flying_grant = witness(
            "Flying grant",
            &[CardType::Enchantment],
            "{1}",
            vec![continuous(
                StaticScope::All(CardPredicate::creature()),
                vec![Modification::AddKeywords(Keywords::of(&[Keyword::Flying]))],
            )],
        );
        let trample_grant = witness(
            "Flying creatures trample",
            &[CardType::Enchantment],
            "{1}",
            vec![continuous(
                StaticScope::All(CardPredicate {
                    any_of: vec![CardPredicate {
                        flying: true,
                        ..Default::default()
                    }],
                    ..CardPredicate::creature()
                }),
                vec![Modification::AddKeywords(Keywords::of(&[Keyword::Trample]))],
            )],
        );
        let grants = if flying_first {
            [flying_grant, trample_grant]
        } else {
            [trample_grant, flying_grant]
        };
        for grant in grants {
            layer_fixture(&mut s, "Gray Ogre", grant);
        }
        let view = s.game().characteristics(ogre);
        assert!(view.keywords.has(Keyword::Flying));
        assert!(view.keywords.has(Keyword::Trample));
    }
}
