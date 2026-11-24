//! Tests for cards filed in `src/dmu.rs`, in card order.

/// Tolarian Terror costs {1} less for each instant and sorcery card in its
/// caster's graveyard, and no less for a creature card there.
#[test]
fn tolarian_terror_costs_less_per_instant_and_sorcery_in_the_graveyard() {
    use crate::support::game::plant_in_graveyard;
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::top_spell,
        play::{cast_card, castable},
    };
    use managym::state::player::ManaPayment;
    let mut s = five_color_unfunded_setup(&[
        "Tolarian Terror",
        "Lightning Strike",
        "Consider",
        "Gray Ogre",
    ]);
    force_lands(&mut s, 0, "Island", 5);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let terror = force_new_card_in_hand(&mut s, 0, "Tolarian Terror");
    plant_in_graveyard(&mut s, 0, "Lightning Strike");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    assert!(!castable(&mut s, terror), "one instant: {{5}}{{U}}");
    plant_in_graveyard(&mut s, 0, "Consider");
    assert!(castable(&mut s, terror), "two spells: {{4}}{{U}}");
    cast_card(&mut s, terror);
    assert_eq!(top_spell(&s).spent.total(), 5);
}

/// Ward {2}: an opponent's spell aimed at Tolarian Terror is countered unless
/// that player pays {2}.
#[test]
fn tolarian_terror_ward_counters_a_spell_unless_two_is_paid() {
    use crate::support::{
        game::{clear_hand, five_color_unfunded_setup, force_lands},
        inspect::permanent,
        play::cast_only,
    };
    use managym::{
        agent::action::{ActionSpaceKind, ActionType},
        flow::turn::StepKind,
        state::game_object::PlayerId,
    };
    for pay in [false, true] {
        let mut s = five_color_unfunded_setup(&["Tolarian Terror", "Lightning Strike"]);
        let terror = s.force_permanent_on_battlefield(0, "Tolarian Terror");
        force_lands(&mut s, 1, "Mountain", 4);
        clear_hand(&mut s, 0);
        clear_hand(&mut s, 1);
        s.force_card_in_hand(1, "Lightning Strike");
        s.advance_to_active_step(1, StepKind::Main);
        cast_only(&mut s);
        s.choose_target_named("Tolarian Terror");
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        assert!(s.take_action_by_type(if pay {
            ActionType::PayCost
        } else {
            ActionType::DeclineChoice
        }));
        s.pass_priority();
        s.pass_priority();
        assert_eq!(permanent(&s, terror).damage, if pay { 3 } else { 0 });
    }
}

/// Garna asks whether the other creature was attacking as it last existed
/// (CR 603.10a): an attacker that dies in combat draws a card; a creature
/// that dies outside combat, or beside Garna in one event, deals 1 damage to
/// each opponent. Both seats see the departed creature's attacking status on
/// the stacked trigger, and the acting seat's text says so.
#[test]
fn garna_draws_for_a_dying_attacker_and_pings_otherwise() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::hand,
        play::{attack_with, bolt, cast, pass_until, settle},
    };
    use managym::{
        agent::{
            action::{Action, ActionSpaceKind},
            observation::Observation,
        },
        state::{game_object::PlayerId, player::ManaPayment},
    };
    let stacked = |s: &crate::support::game::Scenario| {
        [PlayerId(0), PlayerId(1)].map(|viewer| {
            Observation::for_player(s.game(), viewer)
                .continuation
                .into_iter()
                .find(|frame| frame.kind == 3)
                .expect("Garna's trigger")
                .effect_execution
        })
    };
    // An attacker blocked and killed in combat.
    let mut s = five_color_setup(&[
        "Garna, Bloodfist of Keld",
        "Gray Ogre",
        "Hill Giant",
        "Lightning Bolt",
    ]);
    s.force_permanent_on_battlefield(0, "Garna, Bloodfist of Keld");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, ogre);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ogre);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    let block = s
        .action_space()
        .actions
        .iter()
        .position(|a| {
            matches!(a, Action::DeclareBlocker { blocker, attacker: Some(at), .. }
                if *blocker == giant && *at == ogre)
        })
        .unwrap();
    s.step_action(block);
    s.advance_until(
        |s| !s.game().state.stack_objects.is_empty(),
        "Garna's trigger".into(),
    );
    let attacking = stacked(&s);
    assert!(attacking.iter().all(|tokens| !tokens.is_empty()));
    assert_eq!(attacking[0], attacking[1], "public last known information");
    let text = s
        .game()
        .decision_context(PlayerId(0))
        .unwrap()
        .pending
        .join(" ");
    assert!(text.contains("attacking"), "{text}");
    let before = hand(&s, 0).len();
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before + 1);
    s.assert_life(1, 20);

    // A creature that was not attacking.
    let mut s = five_color_setup(&["Garna, Bloodfist of Keld", "Gray Ogre", "Lightning Bolt"]);
    s.force_permanent_on_battlefield(0, "Garna, Bloodfist of Keld");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    bolt(&mut s, ogre);
    let resting = stacked(&s);
    assert_ne!(resting[0], attacking[0]);
    let text = s
        .game()
        .decision_context(PlayerId(0))
        .unwrap()
        .pending
        .join(" ");
    assert!(!text.contains("attacking"), "{text}");
    let before = hand(&s, 0).len();
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before);
    s.assert_life(1, 19);

    // Garna dying beside another creature still sees it die (CR 603.10a),
    // and does not trigger for itself.
    let mut s = five_color_setup(&["Garna, Bloodfist of Keld", "Gray Ogre", "Fumigate"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Garna, Bloodfist of Keld");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Fumigate");
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 1,
        "one Garna trigger".into(),
    );
    settle(&mut s);
    s.assert_life(1, 19);
    s.assert_life(0, 22);
}

#[test]
fn bite_down_binds_both_targets_and_damages_loyalty() {
    use crate::support::game::five_color_setup;
    use crate::support::game::move_card;
    use crate::support::inspect::card_of;
    use crate::support::inspect::permanent_of;
    use crate::support::play::settle;
    use crate::support::{
        game::{force_new_card_in_hand, set_counters},
        inspect::counters_of,
        play::cast_card,
    };
    use managym::state::zone::ZoneType;
    use managym::state::{
        card::{CardType, CardTypes},
        permanent::CounterKind,
        player::ManaPayment,
    };
    let mut s = five_color_setup(&["Bite Down", "Gray Ogre", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let walker = s.force_permanent_on_battlefield(1, "Hill Giant");
    let card = card_of(&s, walker);
    s.game_mut().state[card].types = CardTypes::new([CardType::Planeswalker]);
    set_counters(&mut s, walker, CounterKind::Loyalty, 5);
    let spell = force_new_card_in_hand(&mut s, 0, "Bite Down");
    cast_card(&mut s, spell);
    s.choose_target_named("Gray Ogre");
    s.choose_target_named("Hill Giant");
    let pending = s.game().clone();
    for blink in [None, Some(source), Some(walker)] {
        *s.game_mut() = pending.clone();
        set_counters(&mut s, source, CounterKind::PlusOnePlusOne, 1);
        let mut victim = walker;
        if let Some(id) = blink {
            let card = card_of(&s, id);
            let moved = move_card(&mut s, card, ZoneType::Exile);
            let returned = move_card(&mut s, moved, ZoneType::Battlefield);
            if id == walker {
                victim = permanent_of(&s, returned).unwrap();
                set_counters(&mut s, victim, CounterKind::Loyalty, 5);
            }
        }
        settle(&mut s);
        assert_eq!(
            counters_of(&s, victim, CounterKind::Loyalty),
            if blink.is_some() { 5 } else { 2 }
        );
    }
}
