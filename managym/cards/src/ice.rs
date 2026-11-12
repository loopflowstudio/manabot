//! Ice Age.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PaymentCost, PlayerSelector, Scope,
        TriggerCondition, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mystic_remora(), pyroclasm()]
}

fn pyroclasm() -> CardDefinition {
    let text_box = "Pyroclasm deals 2 damage to each creature.";

    CardDefinition {
        key: "ice.pyroclasm",
        scryfall_id: "88040748-ad76-4b9a-bd4e-87e5980e9816",
        name: "Pyroclasm".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::deal_damage(
            2,
            ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn mystic_remora() -> CardDefinition {
    let text_box = "Cumulative upkeep {1} (At the beginning of your upkeep, put an age counter on this permanent, then sacrifice it unless you pay its upkeep cost for each age counter on it.)\nWhenever an opponent casts a noncreature spell, you may draw a card unless that player pays {4}.";
    let you_may_draw = Effect::Modal {
        modes: vec![
            vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
            vec![],
        ],
    };
    CardDefinition {
        key: "ice.mystic_remora",
        scryfall_id: "58e93dff-b774-4765-b7bd-d3957e42ff4a",
        name: "Mystic Remora".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                effects: vec![
                    Effect::put_counters(ObjectSelector::This, CounterKind::Age, 1),
                    Effect::CumulativeUpkeep {
                        each: ManaCost::parse("{1}"),
                    },
                ],
            },
            // The caster decides whether to pay; unpaid, the caster's
            // opponent (this ability's controller, 1v1) may draw.
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(
                    Actor::Opponent,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::TriggeringPlayer,
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::parse("{4}")),
                        if_paid: vec![],
                        if_declined: vec![Effect::ForPlayer {
                            player: PlayerSelector::TriggeringPlayersOpponents,
                            effects: vec![you_may_draw],
                        }],
                    }],
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
