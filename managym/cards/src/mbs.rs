//! Mirrodin Besieged.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, Scope},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![black_sun_s_zenith()]
}

fn black_sun_s_zenith() -> CardDefinition {
    let text_box = "Put X -1/-1 counters on each creature. Shuffle Black Sun's Zenith into its owner's library.";
    CardDefinition {
        key: "mbs.black_sun_s_zenith",
        scryfall_id: "03bdcf52-50b8-42c0-9665-931d83f5f314",
        name: "Black Sun's Zenith".into(),
        mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::put_counters(
                ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
                CounterKind::MinusOneMinusOne,
                EffectValue::X,
            ),
            Effect::move_to(ObjectSelector::This, ZoneType::Library),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
