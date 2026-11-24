use crate::support::{
    game::five_color_setup,
    inspect::{counters, offered_targets, permanent_named, permanent_of, zone_of},
    play::{cast_owned, enter_and_settle, pass_until, play_and_settle, settle, spend_colorless},
};
use managym::{
    agent::action::ActionSpaceKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

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
