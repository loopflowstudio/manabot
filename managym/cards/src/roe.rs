use managym_state::{
    ability::{Ability, Effect, EffectValue, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        StaticAbility,
    },
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![prophetic_prism(), training_grounds()]
}

fn prophetic_prism() -> CardDefinition {
    let text_box = "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color.";

    CardDefinition {
        key: "roe.prophetic_prism",
        scryfall_id: "cfb90d44-8cb1-4b83-b2f2-92c19d6304fb",
        printed: PrintedCharacteristics {
            name: "Prophetic Prism".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn training_grounds() -> CardDefinition {
    let text_box = "Activated abilities of creatures you control cost {2} less to activate. This effect can't reduce the mana in that cost to less than one mana.";
    CardDefinition {
        key: "roe.training_grounds",
        scryfall_id: "e2cf16f8-6e69-46b3-8453-1d1a2a5670e2",
        printed: PrintedCharacteristics {
            name: "Training Grounds".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::ActivatedAbilitiesCostLess {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                generic: 2,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
