//! Weatherlight.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, StaticAbility, StaticKeywordGrant,
        StaticScope,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fervor()]
}

fn fervor() -> CardDefinition {
    let text_box = "Creatures you control have haste. (They can attack and {T} as soon as they come under your control.)";
    CardDefinition {
        key: "wth.fervor",
        scryfall_id: "b4df70ea-2b6b-4e25-a564-655989ef16fa",
        name: "Fervor".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::YouControl(CardPredicate::creature()),
            condition: None,
            keywords: Keywords::of(&[Keyword::Haste]),
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}
