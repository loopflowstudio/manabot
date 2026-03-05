use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::{mana::Color, zone::ZoneType},
};

use super::helpers::*;

/// CR 106.3 — Mana abilities can be activated to produce mana used for casting.
#[test]
fn cr_106_3_casting_uses_mana_abilities() {
    let mut s = Scenario::new()
        .deck(forest_elves_deck())
        .deck(mountain_deck())
        .seed(3)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    s.assert_action_available(ActionType::PriorityPlayLand);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.assert_action_available(ActionType::PriorityCastSpell);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    assert_eq!(s.zone_size(0, ZoneType::Stack), 1);

    s.pass_priority();
    s.pass_priority();

    assert!(
        !s.battlefield_permanents_named(0, "Llanowar Elves")
            .is_empty(),
        "cast creature should resolve to the battlefield"
    );

    let forest = s
        .battlefield_permanents_named(0, "Forest")
        .into_iter()
        .next()
        .expect("forest should be on battlefield");
    let tapped = s
        .game()
        .state
        .permanents
        .get(forest.0)
        .and_then(|perm| perm.as_ref())
        .map(|perm| perm.tapped)
        .unwrap_or(false);
    assert!(tapped, "land used for mana should be tapped");
}

/// CR 106.4 — Unspent mana empties from players' mana pools as steps end.
#[test]
fn cr_106_4_mana_pool_empties_between_steps() {
    let mut s = Scenario::new()
        .deck(mountain_deck())
        .deck(mountain_deck())
        .seed(9)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "3R");
    assert_eq!(
        s.game().state.players[0].mana_pool.mana[Color::Red as usize],
        1
    );

    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// Negative: a spell is not castable without enough producible mana.
#[test]
fn cr_106_negative_spell_not_available_without_mana() {
    let mut s = Scenario::new()
        .deck(ogre_only_deck())
        .deck(mountain_deck())
        .seed(15)
        .skip_trivial(false)
        .build();

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Grey Ogre");

    s.assert_action_not_available(ActionType::PriorityCastSpell);
}
