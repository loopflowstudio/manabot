//! Commander 2014.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![jazal_goldmane()]
}

fn jazal_goldmane() -> CardDefinition {
    let text_box = "First strike (This creature deals combat damage before creatures without first strike.)\n{3}{W}{W}: Attacking creatures you control get +X/+X until end of turn, where X is the number of attacking creatures.";
    let attacking = CardPredicate {
        attacking: true,
        ..CardPredicate::creature()
    };
    let count = ObjectSelector::each(Scope::Battlefield, attacking.clone()).count();
    CardDefinition {
        key: "c14.jazal_goldmane",
        scryfall_id: "c410d530-e9fc-4dc0-a4bd-70bd70aaf0c7",
        printed: PrintedCharacteristics {
            name: "Jazal Goldmane".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Cat".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{W}{W}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(attacking),
                    Modification::ModifyPt {
                        power: count.clone(),
                        toughness: count,
                    },
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
