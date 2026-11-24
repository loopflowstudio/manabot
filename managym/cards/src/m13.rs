//! Magic 2013.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![krenko_mob_boss()]
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
                    "token.goblin",
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
