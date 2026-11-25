//! Tests for cards filed in `src/m14.rs`, in card order.
use crate::support::game::{five_color_setup, force_new_card_in_hand, Scenario};
use crate::support::inspect::{counters, offered_targets, permanent_named, permanent_of, zone_of};
use crate::support::play::{
    cast_card, cast_owned, enter_and_settle, pass_until, play_and_settle, settle, spend_colorless,
};
use managym::agent::action::ActionSpaceKind;
use managym::flow::turn::StepKind;
use managym::state::game_object::{PlayerId, Target};
use managym::state::player::ManaPayment;
use managym::state::zone::ZoneType;

/// Mindsparker deals 2 damage to an opponent who casts a white or blue
/// instant or sorcery; a red instant, a blue creature and its controller's
/// own spells do nothing.
#[test]
fn mindsparker_burns_opponents_casting_white_or_blue_instants_and_sorceries() {
    let cast_by = |s: &mut Scenario, player: usize, name: &str, target: Option<Target>| {
        let card = force_new_card_in_hand(s, player, name);
        cast_card(s, card);
        if let Some(target) = target {
            assert!(s.choose_target(target));
        }
        settle(s);
    };
    let mut s = five_color_setup(&[
        "Mindsparker",
        "Consider",
        "Scout's Warning",
        "Lightning Bolt",
        "Spectral Sailor",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    s.force_permanent_on_battlefield(0, "Mindsparker");
    cast_by(&mut s, 0, "Consider", None);
    s.assert_life(0, 20);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    cast_by(&mut s, 1, "Consider", None);
    s.assert_life(1, 18);
    cast_by(&mut s, 1, "Scout's Warning", None);
    s.assert_life(1, 16);
    cast_by(
        &mut s,
        1,
        "Lightning Bolt",
        Some(Target::Player(PlayerId(0))),
    );
    cast_by(&mut s, 1, "Spectral Sailor", None);
    s.assert_life(1, 16);
    s.assert_life(0, 17);
}

#[test]
fn token_producer_primeval_bounty() {
    let mut s = five_color_setup(&["Primeval Bounty", "Gray Ogre", "Sol Ring"]);
    enter_and_settle(&mut s, "Primeval Bounty");
    play_and_settle(&mut s, "Gray Ogre");
    let beast = permanent_named(&s, "Beast");
    assert_eq!(s.game().characteristics(beast).pt(), (3, 3));
    let ogre = permanent_named(&s, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast_owned(&mut s, "Sol Ring");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let targets = offered_targets(&s);
    assert_eq!(targets.len(), 2);
    assert!(targets.contains(&Target::Permanent(beast)));
    assert!(targets.contains(&Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(beast)));
    settle(&mut s);
    assert_eq!(counters(&s, beast), 3);
    enter_and_settle(&mut s, "Forest");
    s.assert_life(0, 23);
    assert_eq!(s.battlefield_permanents_named(0, "Beast"), [beast]);
}

#[test]
fn rise_of_the_dark_realms_takes_every_graveyard_creature_under_the_caster() {
    let (s, moved) = crate::return_from_two_graveyards("Rise of the Dark Realms");
    let state = &s.game().state;
    for object in [moved.momo, moved.sailor, moved.drake, moved.their_sailor] {
        let permanent = permanent_of(&s, object).unwrap();
        assert_eq!(
            state.permanents[permanent].as_ref().unwrap().controller,
            PlayerId(0)
        );
    }
    assert_eq!(state[moved.their_sailor].owner, PlayerId(1));
    assert_eq!(zone_of(&s, moved.bolt), Some(ZoneType::Graveyard));
    let momo = permanent_of(&s, moved.momo).unwrap();
    assert_eq!(
        s.game().characteristics(momo).power,
        4,
        "both Sailors and the Drake entered with Momo under one controller"
    );
}
