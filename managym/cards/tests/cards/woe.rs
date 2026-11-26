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
        let mut s = five_color_unfunded_setup(&["Archon's Glory", "Gray Ogre", "Kyoshi Village"]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        // Two distinct white sources, so the mana is a choice and the payment
        // stays pending after the sacrifice is chosen.
        s.force_permanent_on_battlefield(0, "Plains");
        s.force_permanent_on_battlefield(0, "Kyoshi Village");
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
            // The choice is named for bargain, not for a kicker.
            assert_eq!(
                s.game().decision_context(PlayerId(0)).unwrap().instruction,
                "Choose whether to bargain: sacrifice an artifact, enchantment or token as you cast this spell."
            );
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
            // Chosen and still on the battlefield until the mana is paid.
            assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
            let context = s.game().decision_context(PlayerId(0)).unwrap();
            assert_eq!(
                context.paid[0].phrase,
                "Treasure, sacrificed to bargain when the payment completes"
            );
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

/// Which permanent is sacrificed to bargain is the player's choice under
/// automatic payment too: both are offered, each choice is its own line of
/// play that casts the spell once, and the mana is paid for the player.
#[test]
fn archons_glory_bargain_sacrifice_is_chosen_under_auto_and_forks() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_new_card_in_hand},
        inspect::offered_cards,
        play::{cast_card, choose, fork_selecting, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{
            game_object::{PlayerId, Target},
            payment::{How, Toward},
            player::ManaPayment,
            stack_object::StackObject,
        },
    };
    let mut s = five_color_unfunded_setup(&["Archon's Glory", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Plains");
    let tokens = ["token.treasure", "token.food"].map(|key| {
        let token = s.game_mut().create_token(key, PlayerId(0), false).unwrap();
        s.game().state.permanents[token].as_ref().unwrap().object
    });
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let glory = force_new_card_in_hand(&mut s, 0, "Archon's Glory");
    cast_card(&mut s, glory);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    let mut offered = offered_cards(&s);
    offered.sort();
    let mut expected = tokens.to_vec();
    expected.sort();
    assert_eq!(offered, expected);

    for (chosen, kept) in [(tokens[0], tokens[1]), (tokens[1], tokens[0])] {
        let fork = fork_selecting(&s, chosen);
        let [StackObject::Spell(spell)] = fork.game().state.stack_objects.as_slice() else {
            panic!("the spell is cast once, its mana paid automatically")
        };
        assert_eq!(
            spell
                .spent
                .contributed(How::Sacrificed, Toward::Bargain)
                .collect::<Vec<_>>(),
            [chosen]
        );
        assert_eq!(spell.spent.total(), 1);
        assert!(fork.game().state.permanent_of(chosen).is_none());
        assert!(fork.game().state.permanent_of(kept).is_some());
    }
}

/// Mocking Sprite makes its controller's instant and sorcery spells cost {1}
/// less, and no creature spell.
#[test]
fn mocking_sprite_reduces_only_instant_and_sorcery_spells() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::top_spell,
        play::{cast_card, castable, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        player::ManaPayment,
    };
    let mut s = five_color_unfunded_setup(&["Mocking Sprite", "Lightning Strike", "Gray Ogre"]);
    force_lands(&mut s, 0, "Mountain", 2);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let strike = force_new_card_in_hand(&mut s, 0, "Lightning Strike");
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    assert!(!castable(&mut s, ogre));
    s.force_permanent_on_battlefield(0, "Mocking Sprite");
    assert!(!castable(&mut s, ogre), "a creature spell is not reduced");
    assert!(castable(&mut s, strike));
    cast_card(&mut s, strike);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(top_spell(&s).spent.total(), 1);
    settle(&mut s);
    s.assert_life(1, 17);
}

/// Stroke of Midnight's Human goes to the destroyed permanent's controller,
/// and to the controller of an indestructible target that survives.
#[test]
fn stroke_of_midnight_gives_the_targets_controller_a_human() {
    use crate::support::{
        game::{five_color_setup, until_end_of_turn},
        play::{cast, settle},
    };
    use managym::state::{
        card::{Keyword, Keywords},
        effect::Modification,
        game_object::Target,
    };
    for (owner, indestructible) in [(1, false), (0, false), (1, true)] {
        let mut s = five_color_setup(&["Stroke of Midnight", "Gray Ogre"]);
        let ogre = s.force_permanent_on_battlefield(owner, "Gray Ogre");
        if indestructible {
            until_end_of_turn(
                &mut s,
                ogre,
                Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
            );
        }
        cast(&mut s, "Stroke of Midnight");
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        assert_eq!(s.game().state.permanents[ogre].is_some(), indestructible);
        assert_eq!(s.battlefield_permanents_named(owner, "Human").len(), 1);
        assert!(s
            .battlefield_permanents_named(1 - owner, "Human")
            .is_empty());
    }
}

#[test]
fn ruby_daring_tracker_taps_for_either_color() {
    use crate::support::game::five_color_setup;
    use crate::support::play::activate;
    use crate::support::play::choose_mode;
    use managym::state::mana::Mana;
    let mut s = five_color_setup(&["Ruby, Daring Tracker"]);
    let ruby = s.force_permanent_on_battlefield(0, "Ruby, Daring Tracker");
    s.set_player_mana_pool(0, "");
    s.allow_priority_floats();
    activate(&mut s, ruby, 0);
    let choice = s.game().clone();
    assert_eq!(s.action_space().actions.len(), 2);
    for (mode, color) in ["R", "G"].into_iter().enumerate() {
        *s.game_mut() = choice.clone();
        choose_mode(&mut s, mode);
        assert!(crate::support::inspect::tapped(&s, ruby));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse(color));
    }
}

#[test]
fn foundations_5e_quick_study() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::hand;
    use crate::support::play::{cast, settle};
    let mut s = five_color_setup(&["Quick Study"]);
    let before = hand(&s, 0).len();
    cast(&mut s, "Quick Study");
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before + 2);
}
