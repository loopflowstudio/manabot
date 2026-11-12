//! Bloomburrow.
use managym_state::effect::Modification;
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![diresight(), savor()]
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
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(-2, -2),
            ),
            Effect::create_tokens("Food", 1),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn diresight() -> CardDefinition {
    let text_box = "Surveil 2, then draw two cards. You lose 2 life. (To surveil 2, look at the top two cards of your library, then put any number of them into your graveyard and the rest on top of your library in any order.)";
    CardDefinition {
        key: "blb.diresight",
        scryfall_id: "fada29c0-5293-40a4-b36d-d073ee99e650",
        name: "Diresight".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::Surveil { count: 2 },
            Effect::DrawCards {
                count: EffectValue::Fixed(2),
            },
            Effect::LoseLife {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(2),
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
