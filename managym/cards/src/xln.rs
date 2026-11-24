//! Ixalan.
use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, Scope, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        pirate_s_cutlass(),
        dive_down(),
        rivers_rebuke(),
        bishops_soldier(),
    ]
}

fn bishops_soldier() -> CardDefinition {
    let text_box =
        "Lifelink (Damage dealt by this creature also causes you to gain that much life.)";
    CardDefinition {
        key: "xln.bishops_soldier",
        scryfall_id: "d954677c-2de6-440a-90d0-bab2e0c8b4af",
        printed: PrintedCharacteristics {
            name: "Bishop's Soldier".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rivers_rebuke() -> CardDefinition {
    CardDefinition {
        key: "xln.rivers_rebuke",
        scryfall_id: "fda8ef30-bbfa-4857-9750-0dd0def8b13f",
        printed: PrintedCharacteristics {
            name: "River's Rebuke".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::return_to_hand(ObjectSelector::each(
                Scope::TargetPlayerControls,
                CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            ))],
            text_box: "Return all nonland permanents target player controls to their owner's hand."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dive_down() -> CardDefinition {
    let text_box = "Target creature you control gets +0/+3 and gains hexproof until end of turn. (It can't be the target of spells or abilities your opponents control.)";
    CardDefinition {
        key: "xln.dive_down",
        scryfall_id: "b33e493e-1aef-43b3-9716-52158b002430",
        printed: PrintedCharacteristics {
            name: "Dive Down".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(TargetSpec::CreatureYouControl),
                0,
                3,
                Keyword::Hexproof,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pirate_s_cutlass() -> CardDefinition {
    let text_box = "When this Equipment enters, attach it to target Pirate you control.\nEquipped creature gets +2/+1.\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "xln.pirate_s_cutlass",
        scryfall_id: "b3e7e871-19cf-486d-bacb-1499fe066974",
        printed: PrintedCharacteristics {
            name: "Pirate's Cutlass".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::subtype("Pirate"),
                    }),
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::pump(2, 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
