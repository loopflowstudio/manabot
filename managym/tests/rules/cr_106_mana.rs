use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::{mana::Color, zone::ZoneType},
};

use crate::support::game::*;

/// CR 106.3 — Mana abilities can be activated to produce mana used for casting.
#[test]
fn cr_106_3_casting_uses_mana_abilities() {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Mountain", 40)]),
        3,
    );

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
    let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Mountain", 40)]), 9);

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
    let mut s = Scenario::new(deck(&[("Gray Ogre", 40)]), deck(&[("Mountain", 40)]), 15);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Gray Ogre");

    s.assert_action_not_available(ActionType::PriorityCastSpell);
}

/// CR 107.4b/c and 118.3: a generic cost accepts any mana; C requires
/// colorless. Exhaust all 0/1 six-type pools against one fixed pip and 0..2
/// generic. The oracle is integer resource counting, independent of payment.
#[test]
fn audit_generic_and_specific_mana_boundary_matrix() {
    use managym::state::mana::{Mana, ManaCost};

    for bits in 0_u8..64 {
        let mut pool = Mana::default();
        pool.mana = std::array::from_fn(|slot| (bits >> slot) & 1);
        for (slot, symbol) in ["W", "U", "B", "R", "G", "C"].iter().enumerate() {
            for generic in 0..=2 {
                let cost = ManaCost::parse(&format!("{generic}{symbol}"));
                let expected = pool.mana[slot] == 1 && bits.count_ones() > generic;
                assert_eq!(
                    pool.can_pay(&cost),
                    expected,
                    "bits={bits}, cost={generic}{symbol}"
                );
                if expected {
                    let mut remaining = pool.clone();
                    remaining.pay(&cost);
                    assert_eq!(remaining.total(), pool.total() - generic as u16 - 1);
                    assert_eq!(remaining.mana[slot], 0);
                    assert!(remaining
                        .mana
                        .iter()
                        .zip(pool.mana)
                        .all(|(left, before)| *left <= before));
                }
            }
        }
    }
}

/// CR 106.6: spending restrictions survive storage and cannot pay an
/// unrelated cost. Clearing a pool loses both restricted and ordinary mana.
#[test]
fn audit_restricted_mana_never_pays_unrelated_costs() {
    use managym::state::mana::{Mana, ManaCost, ManaUse};

    let lesson = ManaUse::spell(["Lesson"]);
    for slot in 0..6 {
        for eligible in [false, true] {
            let mut amounts = [0; 6];
            amounts[slot] = 1;
            let mut pool = Mana::restricted_to(lesson, amounts);
            let mut cost = ManaCost::parse("1");
            if eligible {
                cost.purpose = lesson;
            }
            assert_eq!(pool.can_pay(&cost), eligible);
            if eligible {
                pool.pay(&cost);
                assert_eq!(pool.total(), 0);
            }
            pool.clear();
            assert_eq!(pool, Mana::default());
        }
    }
}
