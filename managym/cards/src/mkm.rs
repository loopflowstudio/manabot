//! Murders at Karlov Manor.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, TargetSpec,
        TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![make_your_move(), proft_s_eidetic_memory()]
}

fn proft_s_eidetic_memory() -> CardDefinition {
    let text_box = "When Proft's Eidetic Memory enters, draw a card.\nYou have no maximum hand size.\nAt the beginning of combat on your turn, if you've drawn more than one card this turn, put X +1/+1 counters on target creature you control, where X is the number of cards you've drawn this turn minus one.";

    CardDefinition {
        key: "mkm.proft_s_eidetic_memory",
        scryfall_id: "af5b29b3-974c-4200-8df8-b072c11e1600",
        printed: PrintedCharacteristics {
            name: "Proft's Eidetic Memory".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            statics: vec![StaticAbility::NoMaximumHandSize],
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                    intervening_if: Some(Condition::CardsDrawnAtLeast { count: 2 }),
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        EffectValue::CardsDrawnMinus { subtract: 1 },
                    )],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_your_move() -> CardDefinition {
    CardDefinition {
        key: "mkm.make_your_move",
        scryfall_id: "73475d29-2673-4614-86d3-404232426aa8",
        printed: PrintedCharacteristics {
            name: "Make Your Move".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                            CardPredicate {
                                min_power: Some(4),
                                ..CardPredicate::creature()
                            },
                        ],
                        ..Default::default()
                    },
                },
            ))],
            text_box: "Destroy target artifact, enchantment, or creature with power 4 or greater."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
