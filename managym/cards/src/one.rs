//! Phyrexia: All Will Be One.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![offer_immortality()]
}

fn offer_immortality() -> CardDefinition {
    let text_box = "Target creature gains deathtouch and indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";
    CardDefinition {
        key: "one.offer_immortality",
        scryfall_id: "b0aac10a-6d47-4a6c-8a10-2b7c06f3ff32",
        printed: PrintedCharacteristics {
            name: "Offer Immortality".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords::of(&[
                    Keyword::Deathtouch,
                    Keyword::Indestructible,
                ])),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
