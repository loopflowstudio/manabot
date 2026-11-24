//! The Lost Caverns of Ixalan.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![self_reflection()]
}

fn self_reflection() -> CardDefinition {
    CardDefinition {
        key: "lci.self_reflection",
        scryfall_id: "1242203d-c9b5-4ab6-802e-e222f92291e9",
        printed: PrintedCharacteristics {
            name: "Self-Reflection".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::CreateTokenCopy {
                count: 1.into(),
                of: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                exceptions: vec![],
            }],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{U}"))],
            text_box: "Create a token that's a copy of target creature you control.\nFlashback {3}{U} (You may cast this card from your graveyard for its flashback cost. Then exile it.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
