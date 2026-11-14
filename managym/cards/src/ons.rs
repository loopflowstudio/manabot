//! Onslaught.
use managym_state::{
    ability::{Effect, ObjectSelector, PlayerSelector, Scope},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![glory_seeker(), insurrection()]
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
