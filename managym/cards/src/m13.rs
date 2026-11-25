//! Magic 2013.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, LayerCount, LayerValue,
        PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![crusader_of_odric(), krenko_mob_boss(), omniscience()]
}

fn crusader_of_odric() -> CardDefinition {
    let text_box = "Crusader of Odric's power and toughness are each equal to the number of creatures you control.";
    CardDefinition {
        key: "m13.crusader_of_odric",
        scryfall_id: "295096bb-1857-4224-bc7b-307b38cfd338",
        printed: PrintedCharacteristics {
            name: "Crusader of Odric".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            statics: vec![StaticAbility::cda(vec![Modification::SetBasePt {
                power: Some(LayerValue::count(LayerCount::YouControl(
                    CardPredicate::creature(),
                ))),
                toughness: Some(LayerValue::count(LayerCount::YouControl(
                    CardPredicate::creature(),
                ))),
            }])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn krenko_mob_boss() -> CardDefinition {
    let text_box = "{T}: Create X 1/1 red Goblin creature tokens, where X is the number of Goblins you control.";
    CardDefinition {
        key: "m13.krenko_mob_boss",
        scryfall_id: "aa078518-0ce2-4c6f-9061-aa7e22ed7493",
        printed: PrintedCharacteristics {
            name: "Krenko, Mob Boss".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::create_tokens(
                    "token.goblin_1_1_r",
                    ObjectSelector::each_you_control(CardPredicate::subtype("Goblin")).count(),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn omniscience() -> CardDefinition {
    let rules_text = "You may cast spells from your hand without paying their mana costs.";
    CardDefinition {
        key: "m13.omniscience",
        scryfall_id: "1088f33e-cb5f-4248-ae8e-280c4e41f291",
        printed: PrintedCharacteristics {
            name: "Omniscience".into(),
            mana_cost: Some(ManaCost::parse("{7}{U}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::CastFromHandWithoutManaCost],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
