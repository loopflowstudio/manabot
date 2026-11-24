//! Onslaught.
use managym_state::{
    ability::{Actor, Condition, Effect, ObjectSelector, PlayerSelector, Scope},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, DamageTransform, Keyword,
        Keywords, PrintedCharacteristics, ReplacementEffect,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        arcanis_the_omnipotent(),
        glory_seeker(),
        gratuitous_violence(),
        insurrection(),
    ]
}

fn glory_seeker() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "ons.glory_seeker",
        scryfall_id: "9047075e-9fca-484d-bb79-32c0d6821281",
        printed: PrintedCharacteristics {
            name: "Glory Seeker".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gratuitous_violence() -> CardDefinition {
    CardDefinition {
        key: "ons.gratuitous_violence",
        scryfall_id: "4b0c5d14-4fab-4034-a2d3-0d851ef67cbd",
        printed: PrintedCharacteristics {
            name: "Gratuitous Violence".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            replacement_effects: vec![ReplacementEffect::ModifyDamage {
                creature_source: true,
                source_by: Actor::You,
                recipient_by: Actor::AnyPlayer,
                transform: DamageTransform::Double,
            }],
            text_box: "If a creature you control would deal damage to a permanent or player, it deals double that damage instead.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn insurrection() -> CardDefinition {
    let text_box = "Untap all creatures and gain control of them until end of turn. They gain haste until end of turn.";
    CardDefinition {
        key: "ons.insurrection",
        scryfall_id: "998bad32-1927-4e12-9527-efa55b86cae0",
        printed: PrintedCharacteristics {
            name: "Insurrection".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::untap(creatures()),
                Effect::GainControl {
                    player: PlayerSelector::You,
                    what: creatures(),
                    duration: EffectDuration::EndOfTurn,
                },
                Effect::until_end_of_turn(
                    creatures(),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// "All creatures": every creature on the battlefield.
fn creatures() -> ObjectSelector {
    ObjectSelector::each(Scope::Battlefield, CardPredicate::creature())
}

fn arcanis_the_omnipotent() -> CardDefinition {
    CardDefinition {
        key: "ons.arcanis_the_omnipotent",
        scryfall_id: "90865f52-c062-4505-a204-b4d7d4b3fc4c",
        printed: PrintedCharacteristics {
            name: "Arcanis the Omnipotent".into(),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}{U}")),
            power: Some(3),
            toughness: Some(4),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    effects: vec![Effect::DrawCards { count: 3.into() }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}{U}{U}"),
                    // A response can remove Arcanis before this ability resolves.
                    effects: vec![Effect::If {
                        condition: Condition::SourceMatches(CardPredicate::default()),
                        then: vec![Effect::return_to_hand(ObjectSelector::This)],
                        otherwise: vec![],
                    }],
                    ..Default::default()
                },
            ],
            text_box: "{T}: Draw three cards.\n{2}{U}{U}: Return Arcanis to its owner's hand."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
