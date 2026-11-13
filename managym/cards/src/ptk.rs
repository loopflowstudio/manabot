use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        corrupt_court_official(),
        diaochan_artful_beauty(),
        empty_city_ruse(),
        taunting_challenge(),
    ]
}

fn corrupt_court_official() -> CardDefinition {
    let text_box = "When this creature enters, target opponent discards a card.";

    CardDefinition {
        key: "ptk.corrupt_court_official",
        scryfall_id: "9d3ba2e3-e680-47cd-81c5-555deea7d00f",
        printed: PrintedCharacteristics {
            name: "Corrupt Court Official".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Advisor".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Discard {
                    what: ObjectSelector::Target(TargetSpec::Opponent),
                    count: 1,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn diaochan_artful_beauty() -> CardDefinition {
    let text_box = "{T}: Destroy target creature of your choice, then destroy target creature of an opponent's choice. Activate only during your turn, before attackers are declared.";
    let destroy = |opponent_chooses: bool| Effect::TargetClause {
        requirement: TargetRequirement {
            opponent_chooses,
            ..TargetRequirement::one(TargetSpec::Creature)
        },
        effects: vec![Effect::destroy(ObjectSelector::Target(
            TargetSpec::Creature,
        ))],
    };
    CardDefinition {
        key: "ptk.diaochan_artful_beauty",
        scryfall_id: "6180c476-dadf-4c03-ab1e-639386bd4319",
        printed: PrintedCharacteristics {
            name: "Diaochan, Artful Beauty".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Advisor".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                own_turn_only: true,
                before_attackers: true,
                effects: vec![destroy(false), destroy(true)],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn empty_city_ruse() -> CardDefinition {
    let text_box = "Target opponent skips all combat phases of their next turn.";
    CardDefinition {
        key: "ptk.empty_city_ruse",
        scryfall_id: "4d37c84c-80e5-453d-bd2e-4f77ff864c89",
        printed: PrintedCharacteristics {
            name: "Empty City Ruse".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Opponent)],
            spell_effects: vec![Effect::ForPlayer {
                player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                effects: vec![Effect::ApplyContinuous {
                    affected: ObjectSelector::You,
                    modification: Modification::SkipsCombatPhasesNextTurn,
                    duration: EffectDuration::Indefinite,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn taunting_challenge() -> CardDefinition {
    let text_box = "All creatures able to block target creature this turn do so.";
    CardDefinition {
        key: "ptk.taunting_challenge",
        scryfall_id: "4620867c-a3a6-4c81-b923-24007367132e",
        printed: PrintedCharacteristics {
            name: "Taunting Challenge".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::ApplyContinuous {
                affected: ObjectSelector::Target(TargetSpec::Creature),
                modification: Modification::MustBeBlockedByAllAble,
                duration: EffectDuration::EndOfTurn,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
