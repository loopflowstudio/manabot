//! Guilds of Ravnica.
use managym_state::ability::{
    Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition,
    TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
};
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        thousand_year_storm(),
        healer_s_hawk(),
        midnight_reaper(),
        burglar_rat(),
        circuitous_route(),
        swiftblade_vindicator(),
    ]
}

fn midnight_reaper() -> CardDefinition {
    let text_box = "Whenever a nontoken creature you control dies, this creature deals 1 damage to you and you draw a card.";
    CardDefinition {
        key: "grn.midnight_reaper",
        scryfall_id: "5e122fe0-51c5-404d-a7b9-3d161a426c35",
        printed: PrintedCharacteristics {
            name: "Midnight Reaper".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into(), "Knight".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                // Includes its own death (CR 603.10a).
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        ..CardPredicate::creature()
                    }),
                },
                intervening_if: None,
                effects: vec![
                    Effect::deal_damage(1, ObjectSelector::You),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
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

fn thousand_year_storm() -> CardDefinition {
    CardDefinition {
        key: "grn.thousand_year_storm",
        scryfall_id: "270a0863-7d07-43f0-925d-a8ce0383a1cb",
        printed: PrintedCharacteristics {
            name: "Thousand-Year Storm".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::instant_or_sorcery()),
                intervening_if: None,
                effects: vec![Effect::Repeat {
                    count: EffectValue::SpellsCastBeforeTriggering(CardPredicate::instant_or_sorcery()),
                    effect: Box::new(Effect::CopyTriggeringObject),
                }],
            }],
            text_box: "Whenever you cast an instant or sorcery spell, copy it for each other instant and sorcery spell you've cast before it this turn. You may choose new targets for the copies.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
