//! The Lost Caverns of Ixalan.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, PrintedCharacteristics,
        StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Color, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![eaten_by_piranhas(), self_reflection()]
}

fn eaten_by_piranhas() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nEnchant creature\nEnchanted creature loses all abilities and is a black Skeleton creature with base power and toughness 1/1. (It loses all other colors, card types, and creature types.)";
    CardDefinition {
        key: "lci.eaten_by_piranhas",
        scryfall_id: "b0c504ef-2382-4174-9b1d-5f38e12a28fc",
        printed: PrintedCharacteristics {
            name: "Eaten by Piranhas".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            keywords: Keywords::of(&[Keyword::Flash]),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::Attached,
                condition: None,
                characteristic_defining: false,
                modifications: vec![
                    Modification::SetTypes {
                        types: vec![CardType::Creature],
                        subtypes: vec!["Skeleton".into()],
                    },
                    Modification::SetColors([Color::Black].into_iter().collect()),
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

fn self_reflection() -> CardDefinition {
    CardDefinition {
        key: "lci.self_reflection",
        scryfall_id: "1242203d-c9b5-4ab6-802e-e222f92291e9",
        printed: PrintedCharacteristics {
            name: "Self-Reflection".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_token_copies(ObjectSelector::Target(TargetSpec::CreatureYouControl), 1)],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{U}"))],
            text_box: "Create a token that's a copy of target creature you control.\nFlashback {3}{U} (You may cast this card from your graveyard for its flashback cost. Then exile it.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
