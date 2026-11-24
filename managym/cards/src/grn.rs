//! Guilds of Ravnica.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition,
        TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        burglar_rat(),
        circuitous_route(),
        swiftblade_vindicator(),
        healer_s_hawk(),
    ]
}

fn healer_s_hawk() -> CardDefinition {
    let text_box =
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)";

    CardDefinition {
        key: "grn.healer_s_hawk",
        scryfall_id: "3313bd5c-b657-47a3-822a-dd0d9165492a",
        printed: PrintedCharacteristics {
            name: "Healer's Hawk".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn swiftblade_vindicator() -> CardDefinition {
    let text_box = "Double strike (This creature deals both first-strike and regular combat damage.)\nVigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "grn.swiftblade_vindicator",
        scryfall_id: "285c4d9e-0f22-49a8-b68c-150fd0d4b617",
        printed: PrintedCharacteristics {
            name: "Swiftblade Vindicator".into(),
            mana_cost: Some(ManaCost::parse("{R}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample, Keyword::DoubleStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn burglar_rat() -> CardDefinition {
    let text_box = "When this creature enters, each opponent discards a card.";
    CardDefinition {
        key: "grn.burglar_rat",
        scryfall_id: "e9f7f218-fd2a-4233-8753-065ddc314f2d",
        printed: PrintedCharacteristics {
            name: "Burglar Rat".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rat".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::EachOpponent,
                    effects: vec![Effect::discard()],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn circuitous_route() -> CardDefinition {
    let text_box = "Search your library for up to two basic land cards and/or Gate cards, put them onto the battlefield tapped, then shuffle.";
    CardDefinition {
        key: "grn.circuitous_route",
        scryfall_id: "5a970429-6369-4422-a343-00b30267f09d",
        printed: PrintedCharacteristics {
            name: "Circuitous Route".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::search_library(
                CardPredicate {
                    any_of: vec![CardPredicate::basic_land(), CardPredicate::subtype("Gate")],
                    ..Default::default()
                },
                EffectValue::Fixed(2),
                vec![Effect::return_to_battlefield(
                    ObjectSelector::Selected,
                    true,
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
