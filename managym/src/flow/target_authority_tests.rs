//! Tiny rules witnesses for stack-target enumeration, acceptance and resolution.

use std::collections::BTreeMap;

use crate::{
    flow::{game::Game, trigger::PendingTrigger},
    state::{
        ability::{Effect, TargetSpec},
        game_object::{CardId, PlayerId, Target},
        player::PlayerConfig,
        zone::ZoneType,
    },
};

fn fixture() -> (Game, CardId, CardId, Target, Target) {
    let deck = BTreeMap::from([
        ("Gray Ogre".to_string(), 8),
        ("Lightning Bolt".to_string(), 8),
        ("Mountain".to_string(), 24),
    ]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        115_2026,
        false,
    );
    let ogre = game
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    let land = game
        .scenario_force_battlefield(PlayerId(0), "Mountain", true)
        .unwrap();
    let card_named = |name: &str| {
        CardId(
            game.state
                .cards
                .iter()
                .position(|card| card.owner == PlayerId(1) && card.name == name)
                .unwrap(),
        )
    };
    let bolt = card_named("Lightning Bolt");
    let ogre_spell = card_named("Gray Ogre");
    game.push_spell_to_stack(bolt, PlayerId(1), vec![], vec![], false);
    game.push_spell_to_stack(ogre_spell, PlayerId(1), vec![], vec![], false);
    (
        game,
        bolt,
        ogre_spell,
        Target::Permanent(ogre),
        Target::Permanent(land),
    )
}

fn trigger(source: CardId, effects: Vec<Effect>, order: u64) -> PendingTrigger {
    PendingTrigger {
        source_card: source,
        source_ref: None,
        source_lki: None,
        ability_index: 0,
        controller: PlayerId(0),
        targets: vec![],
        target_req_indices: vec![],
        requirement_index: 0,
        enqueue_order: order,
        context: None,
        inline_effects: Some(effects),
    }
}

#[test]
fn stack_target_mana_value_boundaries_have_explicit_expected_choices() {
    let (game, bolt, ogre, permanent, land) = fixture();
    // CR 202.3: Bolt has mana value 1, Ogre 3, Mountain 0. CR 115.1 / 601.2c:
    // a restricted target must satisfy the restriction, including equality.
    // Preserve candidate order: newest spell first, then battlefield order.
    let cases = [
        (
            0,
            vec![
                Target::StackSpell(ogre),
                Target::StackSpell(bolt),
                permanent,
                land,
            ],
        ),
        (
            1,
            vec![
                Target::StackSpell(ogre),
                Target::StackSpell(bolt),
                permanent,
            ],
        ),
        (2, vec![Target::StackSpell(ogre), permanent]),
        (3, vec![Target::StackSpell(ogre), permanent]),
        (4, vec![]),
    ];
    for (minimum, expected) in cases {
        let spec = TargetSpec::SpellOrPermanent {
            min_mana_value: minimum,
        };
        assert_eq!(game.legal_targets_for_spec(&spec, PlayerId(0)), expected);
        for target in [
            Target::StackSpell(ogre),
            Target::StackSpell(bolt),
            permanent,
            land,
        ] {
            assert_eq!(
                game.target_is_legal(target, &spec, PlayerId(0)),
                expected.contains(&target)
            );
        }
    }
    assert_eq!(
        game.legal_targets_for_spec(&TargetSpec::Spell, PlayerId(0)),
        vec![Target::StackSpell(ogre), Target::StackSpell(bolt)]
    );
}

#[test]
fn restricted_trigger_rejects_low_value_spell_and_counters_legal_spell() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![Effect::CounterSpell {
            target: TargetSpec::SpellOrPermanent { min_mana_value: 2 },
        }],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    assert!(game
        .choose_trigger_target(PlayerId(0), Target::StackSpell(bolt))
        .is_err());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    game.resolve_top_of_stack();
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Graveyard));
    assert_eq!(game.state.zones.zone_of(bolt), Some(ZoneType::Stack));
}

#[test]
fn departed_only_target_prevents_even_untargeted_gain_life() {
    let (mut game, bolt, ogre, _, _) = fixture();
    game.state.pending_triggers.push(trigger(
        bolt,
        vec![
            Effect::CounterSpell {
                target: TargetSpec::SpellOrPermanent { min_mana_value: 2 },
            },
            Effect::GainLife { amount: 7 },
        ],
        0,
    ));
    assert!(game.flush_triggers().is_some());
    game.choose_trigger_target(PlayerId(0), Target::StackSpell(ogre))
        .unwrap();
    assert!(game.flush_triggers().is_none());
    game.move_card(ogre, ZoneType::Hand);
    // CR 608.2b: with every target illegal the whole ability fails to resolve,
    // including its untargeted instruction. This is more than skipping countering.
    game.resolve_top_of_stack();
    assert_eq!(game.state.players[0].life, 20);
    assert_eq!(game.state.zones.zone_of(ogre), Some(ZoneType::Hand));
}

#[test]
#[ignore = "CR 603.3b missing same-controller trigger-order decision; proposal only"]
fn audit_same_controller_triggers_require_an_order_choice() {
    let (mut game, bolt, _, _, _) = fixture();
    game.state.pending_triggers = vec![
        trigger(bolt, vec![Effect::GainLife { amount: 2 }], 0),
        trigger(bolt, vec![Effect::DealDamageToController { amount: 2 }], 1),
    ];
    // At 2 life, the controller may stack damage below gain (survive at 2),
    // or gain below damage (lose at 0 before gain resolves): CR 603.3b,
    // 405.2, 117.5, 704.5a. An immediate fixed ordering cannot express both.
    game.state.players[0].life = 2;
    assert!(
        game.flush_triggers().is_some(),
        "must suspend for the controller's order choice"
    );
    assert!(
        game.state.stack_objects.len() == 2,
        "neither waiting trigger should be placed before choosing order"
    );
}

#[test]
fn simultaneous_trigger_orders_have_distinct_rules_outcomes() {
    for (gain_last, expected_life, expected_alive) in [(true, 2, true), (false, 0, false)] {
        let (mut game, source, _, _, _) = fixture();
        game.state.players[0].life = 2;
        let gain = vec![Effect::GainLife { amount: 2 }];
        let damage = vec![Effect::DealDamageToController { amount: 2 }];
        let (first, last) = if gain_last {
            (damage, gain)
        } else {
            (gain, damage)
        };
        game.state.pending_triggers = vec![trigger(source, first, 0), trigger(source, last, 1)];
        assert!(game.flush_triggers().is_none());
        game.resolve_top_of_stack();
        game.perform_state_based_actions();
        if game.state.players[0].alive {
            game.resolve_top_of_stack();
            game.perform_state_based_actions();
        }
        // CR 405.2: last placed resolves first. CR 704.5a: lose at zero
        // before the next ability resolves. Both placements must be choices.
        assert_eq!(game.state.players[0].life, expected_life);
        assert_eq!(game.state.players[0].alive, expected_alive);
    }
}

#[test]
#[ignore = "CR 702.11b SpellOrPermanent omits hexproof; world-change proposal only"]
fn audit_spell_or_permanent_must_respect_opponents_hexproof() {
    let (mut game, _, _, permanent, _) = fixture();
    let Target::Permanent(id) = permanent else {
        unreachable!()
    };
    game.state.permanents[id]
        .as_mut()
        .unwrap()
        .temp_keywords
        .hexproof = true;
    // CR 702.11b applies regardless of the wording used to target the object.
    // A creature-only restriction and a spell-or-permanent restriction must
    // both exclude this opposing Ogre; changing spec cannot bypass hexproof.
    assert!(!game.target_is_legal(permanent, &TargetSpec::Creature, PlayerId(1)));
    let spec = TargetSpec::SpellOrPermanent { min_mana_value: 3 };
    assert!(!game
        .legal_targets_for_spec(&spec, PlayerId(1))
        .contains(&permanent));
    assert!(!game.target_is_legal(permanent, &spec, PlayerId(1)));
}
