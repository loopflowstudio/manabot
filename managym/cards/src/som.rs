//! Scars of Mirrodin.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, Scope},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![exsanguinate(), steel_hellkite()]
}

fn exsanguinate() -> CardDefinition {
    let text_box = "Each opponent loses X life. You gain life equal to the life lost this way.";
    CardDefinition {
        key: "som.exsanguinate",
        scryfall_id: "0878b541-a730-49db-b062-5a01656e269d",
        printed: PrintedCharacteristics {
            name: "Exsanguinate".into(),
            mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: vec![
                Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::X,
                },
                Effect::gain_life(EffectValue::LifeLostThisWay),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn steel_hellkite() -> CardDefinition {
    let text_box = "Flying\n{2}: This creature gets +1/+0 until end of turn.\n{X}: Destroy each nonland permanent with mana value X whose controller was dealt combat damage by this creature this turn. Activate only once each turn.";
    CardDefinition {
        key: "som.steel_hellkite",
        scryfall_id: "b126ee24-9597-4ee8-9c4d-5caed585424a",
        printed: PrintedCharacteristics {
            name: "Steel Hellkite".into(),
            mana_cost: Some(ManaCost::parse("{6}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: text_box.into(),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}"),
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::ModifyPt {
                            power: EffectValue::Fixed(1),
                            toughness: EffectValue::Fixed(0),
                        },
                    )],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{X}"),
                    once_each_turn: true,
                    effects: vec![Effect::destroy(ObjectSelector::Each {
                        of: Scope::DamagedBySourceInCombat,
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                        except: None,
                        max_mana_value: None,
                        exact_mana_value: Some(Box::new(EffectValue::X)),
                    })],
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
