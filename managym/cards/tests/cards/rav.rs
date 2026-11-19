//! Doubling Season replaces creation and effect-caused counter placement;
//! Guardian of Vitu-Ghazi convokes with a creature of two owed colors.
use crate::support::game::move_card;
use crate::support::{
    game::{
        enters_battlefield_with_triggers, five_color_setup, five_color_unfunded_setup,
        force_new_card_in_hand,
    },
    inspect::{card_of, tapped, top_spell},
    play::{cast_card, choose, pass_until, pay_rest, resolve_everything},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        game_object::PlayerId,
        mana::Color,
        payment::{How, Toward},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

#[test]
fn doubling_season_doubles_entry_lore_and_tokens_but_not_precombat_lore() {
    let mut s = five_color_setup(&["Doubling Season", "Leaves from the Vine", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Doubling Season");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let saga = enters_battlefield_with_triggers(&mut s, 0, "Leaves from the Vine");
    assert_eq!(
        s.game().state.permanents[saga]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        2
    );
    s.game_mut().scenario_refresh_priority().unwrap();
    resolve_everything(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 2);
    // Use a second Saga with enough chapters remaining after its doubled entry.
    // Removing Doubling Season while placing then returning it isolates the
    // actual next precombat turn action from entry replacement.
    let season = s.battlefield_permanents_named(0, "Doubling Season")[0];
    let season_card = s.game().state.permanents[season].as_ref().unwrap().object;
    let season_card = move_card(&mut s, season_card, ZoneType::Hand);
    let second = enters_battlefield_with_triggers(&mut s, 0, "Leaves from the Vine");
    s.game_mut().scenario_refresh_priority().unwrap();
    resolve_everything(&mut s);
    move_card(&mut s, season_card, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    let turn = s.game().state.turn.turn_number;
    s.advance_until(
        |s| s.game().state.turn.turn_number > turn && s.game().active_player() == PlayerId(0),
        "next own turn".into(),
    );
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    assert_eq!(
        s.game().state.permanents[second]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        2
    );
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
}

/// A green-white creature convoking for {6}{G}{W} pays either colored pip:
/// the player chooses which when mana can pay the other, and is not asked
/// when only one choice leaves the cost payable. The automatic payer pays
/// mana with mana: with lands for the whole cost it asks nothing, and short
/// of mana the convoker and its color are the player's picks.
#[test]
fn guardian_of_vitu_ghazi_convoker_chooses_which_owed_color_it_pays() {
    for (mountains, plains, auto) in [
        (6, true, false),
        (6, false, false),
        (6, true, true),
        (6, false, true),
        (5, true, true),
    ] {
        let mut s = five_color_unfunded_setup(&["Guardian of Vitu-Ghazi"]);
        let convoker = s.force_permanent_on_battlefield(0, "Guardian of Vitu-Ghazi");
        for _ in 0..mountains {
            s.force_permanent_on_battlefield(0, "Mountain");
        }
        s.force_permanent_on_battlefield(0, "Forest");
        if plains {
            s.force_permanent_on_battlefield(0, "Plains");
        }
        let guardian = force_new_card_in_hand(&mut s, 0, "Guardian of Vitu-Ghazi");
        if auto {
            s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
        }
        if auto && mountains == 6 && plains {
            // Eight lands pay it all.
            cast_card(&mut s, guardian);
            let spell = top_spell(&s);
            assert_eq!(spell.spent.total(), 8);
            assert!(spell.spent.contributions().is_empty());
            assert!(!tapped(&s, convoker));
            continue;
        }
        cast_card(&mut s, guardian);
        pass_until(&mut s, ActionSpaceKind::PayOrNot);
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == convoker),
        );
        assert!(tapped(&s, convoker));
        let paid = if plains {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
            let labels: Vec<_> = s
                .game()
                .semantic_decision_frame()
                .unwrap()
                .offers
                .iter()
                .map(|offer| offer.label.clone())
                .collect();
            assert_eq!(
                labels,
                [
                    "Guardian of Vitu-Ghazi pays {W} (convoke)",
                    "Guardian of Vitu-Ghazi pays {G} (convoke)"
                ]
            );
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            if !auto {
                pay_rest(&mut s);
            }
            Color::Green
        } else {
            // The Forest is the only green source for the other pip, so
            // the Guardian pays {W} and the seven lands pay the rest.
            assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
            Color::White
        };
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 7);
        if mountains == 6 {
            // A spare land: none of the convoked color is spent.
            assert_eq!(spell.spent.amounts()[paid as usize], 0, "plains {plains}");
        }
        assert_eq!(
            spell
                .spent
                .contributed(How::Tapped, Toward::Convoke)
                .collect::<Vec<_>>(),
            [card_of(&s, convoker)]
        );
    }
}
