//! Bloomburrow.
use crate::state::ability::AffectedSpec;
use crate::state::effect::Modification;
use crate::state::{
    ability::{Effect, TargetSpec},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![savor()]
}

fn savor() -> CardDefinition {
    let text_box = "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";

    CardDefinition {
        key: "blb.savor",
        scryfall_id: "1397f689-dca1-4d35-864b-92c5606afb9a",
        name: "Savor".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::pump(-2, -2),
            ),
            Effect::CreateToken {
                token_name: "Food".into(),
                count: 1,
                tapped_and_attacking: false,
                attach_source: false,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
