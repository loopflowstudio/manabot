//! Tests for cards filed in `src/clb.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, force_new_card_in_hand},
    inspect::{card_of, permanent_named, tapped, top_spell},
    play::{cast_card, pay_rest, settle, tap_for_mana},
};
use managym::state::{
    card::Keyword,
    game_object::PlayerId,
    mana::{Color, SpendEffect},
    player::ManaPayment,
};

#[test]
fn carnelian_orb_gives_haste_only_to_a_dragon_its_mana_paid_for() {
    for auto in [false, true] {
        // Six mana for Shivan Dragon from the Orb and five Mountains: the
        // payment has no choice in it, so both settings spend the Orb's mana.
        let mut s = five_color_unfunded_setup(&[
            "Carnelian Orb of Dragonkind",
            "Shivan Dragon",
            "Gray Ogre",
        ]);
        let orb = s.force_permanent_on_battlefield(0, "Carnelian Orb of Dragonkind");
        for _ in 0..5 {
            s.force_permanent_on_battlefield(0, "Mountain");
        }
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let dragon = force_new_card_in_hand(&mut s, 0, "Shivan Dragon");
        cast_card(&mut s, dragon);
        if !auto {
            tap_for_mana(&mut s, orb);
            pay_rest(&mut s);
        }
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 6, "auto={auto}");
        let from_orb: Vec<_> = spell
            .spent
            .mana()
            .iter()
            .filter(|unit| unit.source.unwrap().object == card_of(&s, orb))
            .collect();
        let [unit] = from_orb[..] else {
            panic!("one unit came from the Orb, auto={auto}")
        };
        assert_eq!(unit.color, Color::Red as u8);
        assert!(matches!(
            unit.effect,
            Some(SpendEffect::SpellGainsKeyword {
                keyword: Keyword::Haste,
                ..
            })
        ));
        // The spell has haste on the stack, and the Dragon keeps it this turn.
        assert!(spell.spent.keywords_gained().has(Keyword::Haste));
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert!(
            context
                .pending
                .iter()
                .any(|line| line.contains("it gains haste until end of turn")),
            "{:?}",
            context.pending
        );
        settle(&mut s);
        let dragon = permanent_named(&s, "Shivan Dragon");
        assert!(s
            .game()
            .characteristics(dragon)
            .keywords
            .has(Keyword::Haste));

        // The same forced payment for a non-Dragon: the unit is spent, its
        // rider does not apply, and nothing gains haste.
        let mut s = five_color_unfunded_setup(&["Carnelian Orb of Dragonkind", "Gray Ogre"]);
        let orb = s.force_permanent_on_battlefield(0, "Carnelian Orb of Dragonkind");
        for _ in 0..2 {
            s.force_permanent_on_battlefield(0, "Mountain");
        }
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
        cast_card(&mut s, ogre);
        if !auto {
            tap_for_mana(&mut s, orb);
            pay_rest(&mut s);
        }
        let spell = top_spell(&s);
        assert!(tapped(&s, orb));
        assert_eq!(spell.spent.total(), 3);
        assert!(spell.spent.mana().iter().all(|unit| unit.effect.is_none()));
        settle(&mut s);
        let ogre = permanent_named(&s, "Gray Ogre");
        assert!(!s.game().characteristics(ogre).keywords.has(Keyword::Haste));
    }
}

#[test]
fn carnelian_orb_mana_floated_first_is_a_distinct_unit_the_player_may_keep() {
    // Tapped at priority, the Orb's unit waits in the pool with its rider.
    let mut s =
        five_color_unfunded_setup(&["Carnelian Orb of Dragonkind", "Shivan Dragon", "Gray Ogre"]);
    let orb = s.force_permanent_on_battlefield(0, "Carnelian Orb of Dragonkind");
    let mountains: Vec<_> = (0..3)
        .map(|_| s.force_permanent_on_battlefield(0, "Mountain"))
        .collect();
    s.allow_priority_floats();
    tap_for_mana(&mut s, orb);
    assert!(tapped(&s, orb));

    // Gray Ogre costs three and four red are available. Manual payment taps
    // the Mountains and never has to spend the Orb's unit.
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    cast_card(&mut s, ogre);
    for land in &mountains {
        tap_for_mana(&mut s, *land);
    }
    // No source is left to tap, so the cost is paid from the pool. Which red
    // unit pays is the player's choice: the Orb's is its own class.
    let frame = s.game().semantic_decision_frame().unwrap();
    let labels: Vec<_> = frame
        .offers
        .iter()
        .map(|offer| offer.label.clone())
        .collect();
    assert!(
        labels.iter().any(|label| label.contains("haste")),
        "{labels:?}"
    );
    while s.game().state.stack_objects.is_empty() {
        let frame = s.game().semantic_decision_frame().unwrap();
        let plain = frame
            .offers
            .iter()
            .position(|offer| offer.label.starts_with("Spend") && !offer.label.contains("haste"))
            .unwrap_or_else(|| panic!("an ordinary red unit is offered: {:?}", frame.offers));
        s.step_action(plain);
    }
    let spell = top_spell(&s);
    assert_eq!(spell.spent.total(), 3);
    assert!(spell
        .spent
        .mana()
        .iter()
        .all(|unit| unit.source.unwrap().object != card_of(&s, orb)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
    settle(&mut s);
    let ogre = permanent_named(&s, "Gray Ogre");
    assert!(!s.game().characteristics(ogre).keywords.has(Keyword::Haste));
}

/// Floated beside ordinary red, the Orb's unit is the automatic payer's
/// first choice for a Dragon, which its rider gives haste, and its last for
/// any other spell, which would waste the rider.
#[test]
fn automatic_payment_spends_the_orbs_unit_first_on_a_dragon_and_last_otherwise() {
    for (name, cost, on_receipt) in [("Shivan Dragon", 6, true), ("Gray Ogre", 3, false)] {
        let mut s = five_color_unfunded_setup(&[
            "Carnelian Orb of Dragonkind",
            "Shivan Dragon",
            "Gray Ogre",
        ]);
        let orb = s.force_permanent_on_battlefield(0, "Carnelian Orb of Dragonkind");
        s.allow_priority_floats();
        tap_for_mana(&mut s, orb);
        // Exactly the spell's cost in ordinary red as well: one unit is left.
        s.game_mut().state.players[0].mana_pool.add(
            &managym::state::mana::Mana {
                mana: [0, 0, 0, cost, 0, 0],
                ..Default::default()
            },
            None,
        );
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let card = force_new_card_in_hand(&mut s, 0, name);
        cast_card(&mut s, card);
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), u16::from(cost), "{name}");
        let from_orb = spell.spent.mana().iter().any(|unit| {
            unit.source
                .is_some_and(|origin| origin.object == card_of(&s, orb))
        });
        assert_eq!(from_orb, on_receipt, "{name}");
        let left = &s.game().state.players[0].mana_pool;
        assert_eq!(left.total(), 1, "{name}");
        // The unit left is the Orb's own class unless the Dragon took it.
        assert_eq!(left.classes().is_empty(), on_receipt, "{name}");
    }
}
