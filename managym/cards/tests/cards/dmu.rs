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

#[test]
fn token_producer_resolute_reinforcements() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Resolute Reinforcements"]);
    enter_and_settle(&mut s, "Resolute Reinforcements");
    crate::assert_creature_tokens(
        &s,
        "Soldier",
        1,
        (1, 1),
        &[managym::state::mana::Color::White],
        &[],
    );
}

#[test]
fn printed_creature_magnigoth_sentry() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Magnigoth Sentry", 4, 4, &[Keyword::Reach]);
}

#[test]
fn foundations_permanent_balmor_battlemage_captain() {
    use crate::support::game::five_color_setup;
    use crate::support::play::play_and_settle;
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Balmor, Battlemage Captain", "Gray Ogre", "Scout's Warning"]);
    let source = s.force_permanent_on_battlefield(0, "Balmor, Battlemage Captain");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    play_and_settle(&mut s, "Scout's Warning");
    assert_eq!(s.game().characteristics(source).pt(), (2, 3));
    assert_eq!(s.game().characteristics(ogre).pt(), (3, 2));
    for id in [source, ogre] {
        assert!(s.game().characteristics(id).keywords.has(Keyword::Trample));
    }
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
    assert!(!s
        .game()
        .characteristics(ogre)
        .keywords
        .has(Keyword::Trample));
}

#[test]
fn foundations_discard_pilfer() {
    crate::assert_revealed_discard("Pilfer", &["Gray Ogre", "Lightning Bolt", "Giant Growth"]);
}

#[test]
fn foundations_5d_micromancer() {
    crate::assert_optional_revealed_search(
        "Micromancer",
        &["Lightning Bolt", "Duress"],
        &["Ghitu Lavarunner", "Day of Judgment"],
    );
}
