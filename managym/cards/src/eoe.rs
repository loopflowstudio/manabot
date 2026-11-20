//! Edge of Eternities cards. Station is an activated ability whose cost taps
//! another creature and whose effect reads that creature's power from the
//! activation's receipt; a Spacecraft's "N+" lines are static abilities
//! conditioned on its charge counters.
use managym_state::{
    ability::{Condition, Effect, EffectValue, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticKeywordGrant, StaticPtBuff, StaticScope,
        StaticTypeGrant,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![lumen_class_frigate()]
}

fn lumen_class_frigate() -> CardDefinition {
    let text_box = "Station (Tap another creature you control: Put charge counters equal to its power on this Spacecraft. Station only as a sorcery. It's an artifact creature at 12+.)\n2+ | Other creatures you control get +1/+1.\n12+ | Flying, lifelink";
    let charged = |count| {
        Some(Condition::SourceCountersAtLeast {
            kind: CounterKind::Charge,
            count,
        })
    };
    CardDefinition {
        key: "eoe.lumen_class_frigate",
        scryfall_id: "0cc59b5a-65fa-47cc-8ac7-b7c3f533a782",
        printed: PrintedCharacteristics {
            name: "Lumen-Class Frigate".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Spacecraft".into()],
            power: Some(3),
            toughness: Some(5),
            statics: vec![
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::OtherYouControl(CardPredicate::creature()),
                    condition: charged(2),
                    power: 1,
                    toughness: 1,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::TypeGrant(StaticTypeGrant {
                    scope: StaticScope::This,
                    types: vec![CardType::Creature],
                    subtypes: Vec::new(),
                    condition: charged(12),
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::This,
                    condition: charged(12),
                    keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
                }),
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                station: true,
                sorcery_timing: true,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::Charge,
                    EffectValue::PowerTappedToPay,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
