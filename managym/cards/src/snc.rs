//! Streets of New Capenna.
use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, LayerValue, PrintedCharacteristics, StaticAbility,
        StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Color, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fake_your_own_death(), witness_protection()]
}

fn witness_protection() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature loses all abilities and is a green and white Citizen creature with base power and toughness 1/1 named Legitimate Businessperson. (It loses all other colors, card types, creature types, and names.)";
    CardDefinition {
        key: "snc.witness_protection",
        scryfall_id: "a2be6f2c-8ad0-402d-a7ca-9fe817e83b72",
        printed: PrintedCharacteristics {
            name: "Witness Protection".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::Attached,
                condition: None,
                characteristic_defining: false,
                modifications: vec![
                    Modification::SetName("Legitimate Businessperson".into()),
                    Modification::SetTypes {
                        types: vec![CardType::Creature],
                        subtypes: vec!["Citizen".into()],
                    },
                    Modification::SetColors([Color::Green, Color::White].into_iter().collect()),
                    Modification::LoseAllAbilities,
                    Modification::SetBasePt {
                        power: Some(LayerValue::Fixed(1)),
                        toughness: Some(LayerValue::Fixed(1)),
                    },
                ],
            })],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(TargetSpec::Creature),
            }],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fake_your_own_death() -> CardDefinition {
    let text_box = "Until end of turn, target creature gets +2/+0 and gains \"When this creature dies, return it to the battlefield tapped under its owner's control and you create a Treasure token.\" (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "snc.fake_your_own_death",
        scryfall_id: "b4e117a8-3291-4f9a-ab00-c820c8e2aa00",
        printed: PrintedCharacteristics {
            name: "Fake Your Own Death".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(2, 0),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::GrantTriggered(Box::new(Ability::Triggered {
                        condition: TriggerCondition::Dies {
                            subject: TriggerSubject::This,
                        },
                        intervening_if: None,
                        effects: vec![
                            Effect::return_to_owner(ObjectSelector::This, true, None),
                            Effect::create_tokens("token.treasure", 1),
                        ],
                    })),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
