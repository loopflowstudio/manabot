use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![lightning_strike()]
}

fn lightning_strike() -> CardDefinition {
    let text_box = "Lightning Strike deals 3 damage to any target.";

    CardDefinition {
        key: "ths.lightning_strike",
        scryfall_id: "bbb03f2e-2b92-4aa1-afae-301ed5d151d3",
        name: "Lightning Strike".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::deal_damage(
            3,
            ObjectSelector::Target(TargetSpec::CreatureOrPlayer),
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
