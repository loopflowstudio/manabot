//! Streets of New Capenna.
use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fake_your_own_death()]
}

fn fake_your_own_death() -> CardDefinition {
    let text_box = "Until end of turn, target creature gets +2/+0 and gains \"When this creature dies, return it to the battlefield tapped under its owner's control and you create a Treasure token.\" (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "snc.fake_your_own_death",
        scryfall_id: "b4e117a8-3291-4f9a-ab00-c820c8e2aa00",
        printed: PrintedCharacteristics {
            name: "Fake Your Own Death".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(2, 0),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::GrantTriggered(Box::new(Ability::Triggered {
                        condition: TriggerCondition::Dies {
                            subject: TriggerSubject::This,
                        },
                        intervening_if: None,
                        effects: vec![
                            Effect::return_to_owner(ObjectSelector::This, true, None),
                            Effect::create_tokens("token.treasure", 1),
                        ],
                    })),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
