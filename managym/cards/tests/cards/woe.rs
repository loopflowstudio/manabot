//! Wilds of Eldraine mana filters.
use crate::support::{game::five_color_unfunded_setup, play::command};
use managym::{agent::action::Action, state::mana::Mana};

/// A non-tapping filter consumes only its once-each-turn use and its {1}.
#[test]
fn scarecrow_guide_filters_without_a_tap_once_each_turn() {
    let mut s = five_color_unfunded_setup(&["Scarecrow Guide"]);
    s.allow_priority_floats();
    let guide = s.force_permanent_on_battlefield(0, "Scarecrow Guide");
    s.set_player_mana_pool(0, "CC");
    s.game_mut().scenario_refresh_priority().unwrap();
    command(
        &mut s,
        |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide),
    );
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("C"));
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 2, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("BC"));
    assert!(!s.game().state.permanents[guide].as_ref().unwrap().tapped);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide)));
}

/// Archon's Glory gives +2/+2, and flying and lifelink only when a
/// permanent was sacrificed to bargain. Bargain is offered only with an
/// artifact, enchantment or token to sacrifice, and the receipt names it.
#[test]
fn archons_glory_gains_flying_and_lifelink_only_when_bargained() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::{has_action, offered_cards, top_spell},
        play::{cast_card, choose, pass_until, pay_rest, select_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{
            card::Keyword,
            game_object::{PlayerId, Target},
            payment::{How, Toward},
        },
    };
    for (treasure, bargain) in [(false, false), (true, false), (true, true)] {
        let mut s = five_color_unfunded_setup(&["Archon's Glory", "Gray Ogre"]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        s.force_permanent_on_battlefield(0, "Plains");
        let token = treasure.then(|| {
            let token = s
                .game_mut()
                .create_token("token.treasure", PlayerId(0), false)
                .unwrap();
            s.game().state.permanents[token].as_ref().unwrap().object
        });
        let glory = force_new_card_in_hand(&mut s, 0, "Archon's Glory");
        cast_card(&mut s, glory);
        // A creature is not an artifact, enchantment or token.
        assert_eq!(
            has_action(&s, |a| matches!(a, Action::Decline { .. })),
            treasure,
            "bargain is offered only with something to sacrifice"
        );
        if treasure {
            assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
            choose(&mut s, |a| {
                if bargain {
                    matches!(a, Action::PayCost { .. })
                } else {
                    matches!(a, Action::Decline { .. })
                }
            });
        }
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(ogre)));
        if bargain {
            let token = token.unwrap();
            assert_eq!(offered_cards(&s), [token]);
            let frame = s.game().semantic_decision_frame().unwrap();
            assert_eq!(frame.offers[0].label, "Sacrifice Treasure (bargain)");
            select_card(&mut s, token);
        }
        if s.action_space().kind == ActionSpaceKind::PayOrNot {
            pay_rest(&mut s);
        }
        let spell = top_spell(&s);
        assert_eq!(spell.spent.total(), 1);
        assert_eq!(
            spell
                .spent
                .contributed(How::Sacrificed, Toward::Bargain)
                .collect::<Vec<_>>(),
            if bargain {
                vec![token.unwrap()]
            } else {
                vec![]
            }
        );
        let context = s.game().decision_context(PlayerId(0)).unwrap();
        assert_eq!(
            context
                .pending
                .iter()
                .any(|line| line.contains("sacrificed Treasure to bargain")),
            bargain,
            "{:?}",
            context.pending
        );
        settle(&mut s);
        assert_eq!(
            s.battlefield_permanents_named(0, "Treasure").len(),
            usize::from(treasure && !bargain)
        );
        let view = s.game().characteristics(ogre);
        assert_eq!((view.power, view.toughness), (4, 4));
        for keyword in [Keyword::Flying, Keyword::Lifelink] {
            assert_eq!(view.keywords.has(keyword), bargain);
        }
    }
}
