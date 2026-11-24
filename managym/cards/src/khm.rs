//! Kaldheim.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, SacrificeCost, StaticAbility,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        fynn_the_fangbearer(),
        koma_cosmos_serpent(),
        seize_the_spoils(),
    ]
}

fn koma_cosmos_serpent() -> CardDefinition {
    let text_box = "This spell can't be countered.\nAt the beginning of each upkeep, create a 3/3 blue Serpent creature token named Koma's Coil.\nSacrifice another Serpent: Choose one —\n• Tap target permanent. Its activated abilities can't be activated this turn.\n• Koma gains indestructible until end of turn.";
    let target = TargetSpec::Permanent {
        predicate: CardPredicate::default(),
    };
    // "Choose one" of an activated ability is chosen as it is activated
    // (CR 700.2a), so each mode is its own ability with the shared cost.
    let mode = |effects: Vec<Effect>| ActivatedAbilityDefinition {
        sacrifice: Some(SacrificeCost {
            predicate: CardPredicate::subtype("Serpent"),
            another: true,
            or_mana: None,
        }),
        effects,
        ..Default::default()
    };
    CardDefinition {
        key: "khm.koma_cosmos_serpent",
        scryfall_id: "2de25ea4-284a-4c16-b823-048ff00c6a03",
        printed: PrintedCharacteristics {
            name: "Koma, Cosmos Serpent".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Serpent".into()],
            power: Some(6),
            toughness: Some(6),
            statics: vec![StaticAbility::CantBeCountered],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::AnyPlayer),
                effects: vec![Effect::create_tokens("token.komas_coil", 1)],
            }],
            activated_abilities: vec![
                mode(vec![
                    Effect::Tap {
                        what: ObjectSelector::Target(target.clone()),
                    },
                    Effect::ApplyContinuous {
                        affected: ObjectSelector::Target(target),
                        modification: Modification::CantActivateAbilities,
                        duration: EffectDuration::EndOfTurn,
                    },
                ]),
                mode(vec![Effect::ApplyContinuous {
                    affected: ObjectSelector::This,
                    modification: Modification::AddKeywords(Keywords::of(&[
                        Keyword::Indestructible,
                    ])),
                    duration: EffectDuration::EndOfTurn,
                }]),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn seize_the_spoils() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, discard a card.\nDraw two cards and create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "khm.seize_the_spoils",
        scryfall_id: "b3b7a69c-75d2-49a6-ab56-ef608d0b0208",
        printed: PrintedCharacteristics {
            name: "Seize the Spoils".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::Discard],
            spell_effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                },
                Effect::create_tokens("token.treasure", 1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fynn_the_fangbearer() -> CardDefinition {
    CardDefinition {
        key: "khm.fynn_the_fangbearer",
        scryfall_id: "7d7a8a90-13c1-4b0c-ab2e-fc8d91ccefd9",
        printed: PrintedCharacteristics {
            name: "Fynn, the Fangbearer".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(1),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            supertypes: vec!["Legendary".into()],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: managym_state::ability::TriggerSubject::you_control(CardPredicate { deathtouch: true, ..CardPredicate::creature() }),
                    to: Actor::Opponent,
                    per: managym_state::ability::Per::Object,
                },
                effects: vec![Effect::GivePlayerCounters {
                    player: managym_state::ability::PlayerSelector::EachOpponent,
                    kind: managym_state::permanent::CounterKind::Poison, count: 2,
                }],
            }],
            text_box: "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)\nWhenever a creature you control with deathtouch deals combat damage to a player, that player gets two poison counters. (A player with ten or more poison counters loses the game.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
