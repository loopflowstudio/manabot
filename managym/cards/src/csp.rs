//! Coldsnap.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::{Color, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![deathmark(), flashfreeze(), dark_depths()]
}

fn dark_depths() -> CardDefinition {
    let text_box = "Dark Depths enters with ten ice counters on it.\n{3}: Remove an ice counter from Dark Depths.\nWhen Dark Depths has no ice counters on it, sacrifice it. If you do, create Marit Lage, a legendary 20/20 black Avatar creature token with flying and indestructible.";
    CardDefinition {
        key: "csp.dark_depths",
        scryfall_id: "92409c3a-fb1a-4205-9fe1-0f5affc7b21d",
        printed: PrintedCharacteristics {
            name: "Dark Depths".into(),
            types: CardTypes::new([CardType::Land]),
            supertypes: vec!["Legendary".into(), "Snow".into()],
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                if_cast: false,
                kind: CounterKind::Ice,
                count: 10,
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}"),
                effects: vec![Effect::RemoveCounters {
                    from: ObjectSelector::This,
                    up_to: 1,
                    chosen: Some(CounterKind::Ice),
                }],
                ..Default::default()
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::NoCounters {
                    kind: CounterKind::Ice,
                },
                effects: vec![
                    Effect::sacrifice_this(),
                    Effect::if_you_do(vec![Effect::create_tokens("token.marit_lage", 1)]),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn flashfreeze() -> CardDefinition {
    CardDefinition {
        key: "csp.flashfreeze",
        scryfall_id: "cefd9955-a195-4855-a00e-3809b96ca92b",
        printed: PrintedCharacteristics {
            name: "Flashfreeze".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellMatching {
                    predicate: CardPredicate {
                        colors_any: vec![Color::Red, Color::Green],
                        ..Default::default()
                    },
                }),
                exile: false,
            }],
            text_box: "Counter target red or green spell.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deathmark() -> CardDefinition {
    let text_box = "Destroy target green or white creature.";
    CardDefinition {
        key: "csp.deathmark",
        scryfall_id: "e72e8728-d0a0-4ee5-87c3-092ca94225e0",
        printed: PrintedCharacteristics {
            name: "Deathmark".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        colors_any: vec![Color::Green, Color::White],
                        ..CardPredicate::creature()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
