//! Adventures in the Forgotten Realms.
use crate::state::{
    ability::{Effect, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, SacrificeCost},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![deadly_dispute(), power_word_kill()]
}

fn deadly_dispute() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, sacrifice an artifact or creature.\nDraw two cards and create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    CardDefinition {
        key: "afr.deadly_dispute",
        scryfall_id: "7373fe95-ad1c-44b9-8c7f-464ce8cbffc6",
        name: "Deadly Dispute".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        sacrifice_cost: Some(SacrificeCost {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Artifact, CardType::Creature],
                ..Default::default()
            },
            another: false,
            or_mana: None,
        }),
        spell_effects: vec![
            Effect::DrawCards { count: 2 },
            Effect::CreateToken {
                token_name: "Treasure".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn power_word_kill() -> CardDefinition {
    let text_box = "Destroy target non-Angel, non-Demon, non-Devil, non-Dragon creature.";

    CardDefinition {
        key: "afr.power_word_kill",
        scryfall_id: "395b6ce4-143f-4eed-b565-98aa3d6208ef",
        name: "Power Word Kill".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::Destroy {
            target: TargetSpec::Permanent {
                predicate: CardPredicate {
                    not_subtypes: vec![
                        "Angel".into(),
                        "Demon".into(),
                        "Devil".into(),
                        "Dragon".into(),
                    ],
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
            },
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
