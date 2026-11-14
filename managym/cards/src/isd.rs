//! Innistrad.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![blasphemous_act(), typhoid_rats()]
}

fn typhoid_rats() -> CardDefinition {
    let text_box =
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)";

    CardDefinition {
        key: "isd.typhoid_rats",
        scryfall_id: "4490ce65-c73a-4809-abd1-ccc3175bd2a4",
        printed: PrintedCharacteristics {
            name: "Typhoid Rats".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rat".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn blasphemous_act() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast for each creature on the battlefield.\nBlasphemous Act deals 13 damage to each creature.";
    CardDefinition {
        key: "isd.blasphemous_act",
        scryfall_id: "509ce648-fb76-486d-8b39-183e368b7cb7",
        printed: PrintedCharacteristics {
            name: "Blasphemous Act".into(),
            mana_cost: Some(ManaCost::parse("{8}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            statics: vec![StaticAbility::CostsLessPerPermanentOnBattlefield(
                CardPredicate::creature(),
            )],
            spell_effects: vec![Effect::deal_damage(
                13,
                ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
