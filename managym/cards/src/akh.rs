//! Amonkhet.
use managym_state::ability::{Ability, Effect, TriggerSubject};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PowerCda, PrintedCharacteristics,
    StaticAbility, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![enigma_drake(), regal_caracal(), vizier_of_the_menagerie()]
}

fn enigma_drake() -> CardDefinition {
    let text_box = "Flying\nEnigma Drake's power is equal to the number of instant and sorcery cards in your graveyard.";
    CardDefinition {
        key: "akh.enigma_drake",
        scryfall_id: "66286631-c16e-410c-b963-25cfe8005d8f",
        printed: PrintedCharacteristics {
            name: "Enigma Drake".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Drake".into()],
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::CharacteristicPower(
                PowerCda::GraveyardMatching(Box::new(CardPredicate {
                    card_types_any: vec![CardType::Instant, CardType::Sorcery],
                    ..Default::default()
                })),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn regal_caracal() -> CardDefinition {
    let text_box = "Other Cats you control get +1/+1 and have lifelink.\nWhen this creature enters, create two 1/1 white Cat creature tokens with lifelink.";
    CardDefinition {
        key: "akh.regal_caracal",
        scryfall_id: "cd349f95-3eae-4ef2-abf8-e911bb8e93e5",
        printed: PrintedCharacteristics {
            name: "Regal Caracal".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(3),
            toughness: Some(3),
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(CardPredicate::subtype("Cat")),
                None,
                vec![
                    Modification::pump(1, 1),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink])),
                ],
            )],
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.cat_1_1_w_lifelink", 2)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vizier_of_the_menagerie() -> CardDefinition {
    CardDefinition {
        key: "akh.vizier_of_the_menagerie",
        scryfall_id: "ca204351-7a7e-4e4b-8c2b-f90fa0f9d724",
        printed: PrintedCharacteristics {
            name: "Vizier of the Menagerie".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Snake".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(4),
            statics: vec![
                StaticAbility::LookAtLibraryTop,
                StaticAbility::CastFromLibraryTop(CardPredicate::creature()),
                StaticAbility::SpendAnyManaToCast(CardPredicate::creature()),
            ],
            text_box: "You may look at the top card of your library any time.\nYou may cast creature spells from the top of your library.\nYou can spend mana of any type to cast creature spells.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
