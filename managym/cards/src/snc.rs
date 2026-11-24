//! Streets of New Capenna.

use managym_state::{
    ability::{Effect, ObjectSelector, PlayerSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::{EffectDuration, Modification},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![involuntary_employment()]
}

fn involuntary_employment() -> CardDefinition {
    let text_box = "Gain control of target creature until end of turn. Untap that creature. It gains haste until end of turn. Create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "snc.involuntary_employment",
        scryfall_id: "af7626e2-0041-4ac5-97b3-2db8dcb094d4",
        printed: PrintedCharacteristics {
            name: "Involuntary Employment".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::GainControl {
                    player: PlayerSelector::You,
                    what: ObjectSelector::Target(TargetSpec::Creature),
                    duration: EffectDuration::EndOfTurn,
                },
                Effect::untap(ObjectSelector::Target(TargetSpec::Creature)),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ),
                Effect::create_tokens("token.treasure", 1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
