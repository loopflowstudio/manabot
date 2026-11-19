//! Wilds of Eldraine.
use managym_state::{
    ability::{Condition, Effect, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, SpellCostReduction, StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![archons_glory(), mocking_sprite(), scarecrow_guide()]
}

fn scarecrow_guide() -> CardDefinition {
    let text_box = "Reach\n{1}: Add one mana of any color. Activate only once each turn.";
    CardDefinition {
        key: "woe.scarecrow_guide",
        scryfall_id: "3dc08461-bec6-4a18-b782-da4c92abb789",
        printed: PrintedCharacteristics {
            name: "Scarecrow Guide".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Scarecrow".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Reach]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                once_each_turn: true,
                mana_cost: ManaCost::parse("1"),
                effects: vec![Effect::any_color_mana()],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn archons_glory() -> CardDefinition {
    let text_box = "Bargain (You may sacrifice an artifact, enchantment, or token as you cast this spell.)\nTarget creature gets +2/+2 until end of turn. If this spell was bargained, that creature also gains flying and lifelink until end of turn.";
    let creature = || ObjectSelector::Target(TargetSpec::Creature);
    CardDefinition {
        key: "woe.archons_glory",
        scryfall_id: "e71768e7-4ef7-4fb2-838b-eb3a7f662d38",
        printed: PrintedCharacteristics {
            name: "Archon's Glory".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Bargain],
            spell_effects: vec![
                Effect::until_end_of_turn(creature(), Modification::pump(2, 2)),
                Effect::If {
                    condition: Condition::Kicked,
                    then: vec![Effect::until_end_of_turn(
                        creature(),
                        Modification::AddKeywords(Keywords::of(&[
                            Keyword::Flying,
                            Keyword::Lifelink,
                        ])),
                    )],
                    otherwise: vec![],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mocking_sprite() -> CardDefinition {
    let text_box = "Flying\nInstant and sorcery spells you cast cost {1} less to cast.";
    CardDefinition {
        key: "woe.mocking_sprite",
        scryfall_id: "e595014d-4ff4-4561-b7f2-a9bd56300b01",
        printed: PrintedCharacteristics {
            name: "Mocking Sprite".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Faerie".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Instant, CardType::Sorcery],
                    ..Default::default()
                },
                condition: Condition::Always,
                generic: 1,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
