//! Salvat 2005.
use managym_state::{
    ability::{
        Ability, DistinctSearchEnd, Effect, ObjectSelector, PaymentCost, Per, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![frenzied_goblin(), three_dreams()]
}

fn frenzied_goblin() -> CardDefinition {
    let text_box =
        "Whenever this creature attacks, you may pay {R}. If you do, target creature can't block this turn.";
    CardDefinition {
        key: "psal.frenzied_goblin",
        scryfall_id: "f786de73-98ff-4426-abca-dc789f6848ce",
        printed: PrintedCharacteristics {
            name: "Frenzied Goblin".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Berserker".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Creature),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::parse("{R}")),
                        if_paid: vec![Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::Creature),
                            Modification::CantBlock,
                        )],
                        if_declined: vec![],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn three_dreams() -> CardDefinition {
    let text_box = "Search your library for up to three Aura cards with different names, reveal them, put them into your hand, then shuffle.";
    CardDefinition {
        key: "psal.three_dreams",
        scryfall_id: "ddd4bbfb-80c9-4bdd-86c6-96fba00d3d12",
        printed: PrintedCharacteristics {
            name: "Three Dreams".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::SearchDistinctNames {
                predicate: CardPredicate::subtype("Aura"),
                count: 3,
                end: DistinctSearchEnd::Hand,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
