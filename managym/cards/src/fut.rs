//! Future Sight.
use managym_state::effect::Modification;
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![scout_s_warning()]
}

fn scout_s_warning() -> CardDefinition {
    let text_box =
        "The next creature card you play this turn can be played as though it had flash.\nDraw a card.";
    CardDefinition {
        key: "fut.scout_s_warning",
        scryfall_id: "e6639e3d-b0c4-4c08-83d5-f3fc58fac9b2",
        name: "Scout's Warning".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::until_end_of_turn(ObjectSelector::You, Modification::NextCreatureCastFlash),
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
