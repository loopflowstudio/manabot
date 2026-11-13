//! Coldsnap.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TriggerCondition},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dark_depths()]
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
