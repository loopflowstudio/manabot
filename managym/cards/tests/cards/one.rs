//! Phyrexia: All Will Be One.

#[test]
fn foundations_5e_offer_immortality() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Offer Immortality",
        "Gray Ogre",
        0,
        (2, 2),
        &[Keyword::Deathtouch, Keyword::Indestructible],
        0,
        0,
    );
}

#[test]
fn ovika_composite_ward_and_haste_tokens() {
    use crate::support::game::{five_color_setup, move_to_hand};
    use crate::support::inspect::{has_action, keywords, permanent};
    use crate::support::play::{cast_card, choose, pass_until, settle, spend_colorless};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::{PlayerId, Target};

    // 1. Ward — {3}, Pay 3 life: unpayable at 2 life; payable at 10 life costs both {3} and 3 life.
    for (foe_life, can_pay) in [(2, false), (10, true)] {
        let mut s = five_color_setup(&["Ovika, Enigma Goliath", "Lightning Bolt"]);
        let ovika = s.force_permanent_on_battlefield(0, "Ovika, Enigma Goliath");
        s.game_mut().state.players[1].life = foe_life;
        s.set_player_mana_pool(1, "3R");
        let bolt = crate::support::game::force_new_card_in_hand(&mut s, 1, "Lightning Bolt");
        // Pass priority to opponent so they can target Ovika with Lightning Bolt.
        s.pass_priority();
        cast_card(&mut s, bolt);
        assert!(s.choose_target(Target::Permanent(ovika)));
        pass_until(&mut s, ActionSpaceKind::PayOrNot);
        assert_eq!(
            has_action(
                &s,
                |a| matches!(a, Action::PayCost { player } if *player == PlayerId(1))
            ),
            can_pay
        );
        if can_pay {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
            spend_colorless(&mut s);
            settle(&mut s);
            s.assert_life(1, foe_life - 3);
            assert_eq!(permanent(&s, ovika).damage, 3);
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
            settle(&mut s);
            s.assert_life(1, foe_life);
            assert_eq!(permanent(&s, ovika).damage, 0);
        }
    }

    // 2. Noncreature cast trigger creates MV 1/1 Phyrexian Goblins with haste until end of turn.
    let mut s = five_color_setup(&["Ovika, Enigma Goliath", "Seismic Rupture", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Ovika, Enigma Goliath");
    let ogre = move_to_hand(&mut s, "Gray Ogre");
    cast_card(&mut s, ogre);
    spend_colorless(&mut s);
    settle(&mut s);
    assert!(s
        .battlefield_permanents_named(0, "Phyrexian Goblin")
        .is_empty());

    let rupture = move_to_hand(&mut s, "Seismic Rupture");
    cast_card(&mut s, rupture);
    spend_colorless(&mut s);
    // Ovika's trigger resolves BEFORE Seismic Rupture: verify 3 1/1 hasty Phyrexian Goblins exist!
    crate::support::play::resolve_top(&mut s);
    let goblins = s.battlefield_permanents_named(0, "Phyrexian Goblin");
    assert_eq!(goblins.len(), 3);
    for g in &goblins {
        assert_eq!(s.game().characteristics(*g).pt(), (1, 1));
        assert!(keywords(&s, *g).has(Keyword::Haste));
    }
    // Also test with a non-damaging 3-MV noncreature (Cancel/Diresight) surviving until next turn!
    let mut s = five_color_setup(&["Ovika, Enigma Goliath", "Diresight"]);
    s.force_permanent_on_battlefield(0, "Ovika, Enigma Goliath");
    let diresight = move_to_hand(&mut s, "Diresight");
    cast_card(&mut s, diresight);
    spend_colorless(&mut s);
    settle(&mut s);
    let goblins = s.battlefield_permanents_named(0, "Phyrexian Goblin");
    assert_eq!(goblins.len(), 3);
    for g in &goblins {
        assert!(keywords(&s, *g).has(Keyword::Haste));
    }
    s.advance_to_active_step(1, StepKind::Upkeep);
    for g in &goblins {
        assert!(!keywords(&s, *g).has(Keyword::Haste));
    }
}
