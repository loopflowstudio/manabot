//! Streets of New Capenna.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![venom_connoisseur()]
}

fn venom_connoisseur() -> CardDefinition {
    let grant = |who| {
        Effect::until_end_of_turn(
            who,
            Modification::AddKeywords(Keywords::of(&[Keyword::Deathtouch])),
        )
    };
    let text_box = "Alliance — Whenever another creature you control enters, this creature gains deathtouch until end of turn. If this is the second time this ability has resolved this turn, all creatures you control gain deathtouch until end of turn.";
    CardDefinition {
        key: "snc.venom_connoisseur",
        scryfall_id: "04897d8c-ee05-45eb-80f7-76487dbcc449",
        printed: PrintedCharacteristics {
            name: "Venom Connoisseur".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Druid".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                effects: vec![
                    grant(ObjectSelector::This),
                    Effect::OnNthResolutionThisTurn {
                        n: 2,
                        effect: Box::new(grant(ObjectSelector::each_you_control(
                            CardPredicate::creature(),
                        ))),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
