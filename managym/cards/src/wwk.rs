//! Worldwake.
use managym_state::{
    ability::{Ability, Effect, Per, PlayerSelector, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![pulse_tracker(), basilisk_collar()]
}

fn basilisk_collar() -> CardDefinition {
    let text_box = "Equipped creature has deathtouch and lifelink. (Any amount of damage it deals to a creature is enough to destroy it. Damage dealt by this creature also causes you to gain that much life.)\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "wwk.basilisk_collar",
        scryfall_id: "55cdba1b-7a80-435f-9cff-b9365f62e311",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::Deathtouch,
                    Keyword::Lifelink,
                ]))],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            name: "Basilisk Collar".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            mana_cost: Some(ManaCost::parse("{1}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pulse_tracker() -> CardDefinition {
    let text_box = "Whenever this creature attacks, each opponent loses 1 life.";
    CardDefinition {
        key: "wwk.pulse_tracker",
        scryfall_id: "4604a63c-ebe0-420f-968e-3ffc7641ce22",
        printed: PrintedCharacteristics {
            name: "Pulse Tracker".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Rogue".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: 1.into(),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
