//! Oath of the Gatewatch.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![untamed_hunger(), make_a_stand(), tajuru_pathwarden()]
}

fn tajuru_pathwarden() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "ogw.tajuru_pathwarden",
        scryfall_id: "b073e75b-b432-41a5-a71e-d169fecf774f",
        printed: PrintedCharacteristics {
            name: "Tajuru Pathwarden".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into(), "Ally".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_a_stand() -> CardDefinition {
    let text_box = "Creatures you control get +1/+0 and gain indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy them.)";
    CardDefinition {
        key: "ogw.make_a_stand",
        scryfall_id: "30cace63-91ca-493c-b67f-740fbbf06370",
        printed: PrintedCharacteristics {
            name: "Make a Stand".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::each_you_control(CardPredicate::creature()),
                1,
                0,
                Keyword::Indestructible,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn untamed_hunger() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature gets +2/+1 and has menace. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "ogw.untamed_hunger",
        scryfall_id: "595deb86-b6cf-4e4f-a6cf-f5ff128b720d",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![
                    Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                    Modification::pump(2, 1),
                ],
            )],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            name: "Untamed Hunger".into(),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
