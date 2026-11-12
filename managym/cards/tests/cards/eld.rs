//! Tests for cards filed in `src/eld.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, permanent_named},
    play::{activate_first, pass_until, play_and_settle, select_card, settle},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = five_color_setup(&[name]);
        play_and_settle(&mut s, name);
        let source = permanent_named(&s, name);
        activate_first(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = library_card(&s, 0, "Forest");
        select_card(&mut s, found);
        settle(&mut s);
        let id = s.game().state.card_to_permanent[found].unwrap();
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().tapped,
            tapped
        );
        assert_eq!(s.game().state.players[0].life, life);
        assert!(s.battlefield_permanents_named(0, name).is_empty());
    }
}

#[test]
fn return_of_the_wildspeaker_reads_and_pumps_only_non_humans() {
    use crate::support::{
        game::until_end_of_turn,
        play::{cast, choose_mode},
    };
    use managym::state::{effect::Modification, zone::ZoneType};
    let setup = || {
        let mut s = five_color_setup(&[
            "Return of the Wildspeaker",
            "Gray Ogre",
            "Agent of Treachery",
        ]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let agent = s.force_permanent_on_battlefield(0, "Agent of Treachery");
        // The Human is the strongest creature, and must not count.
        until_end_of_turn(
            &mut s,
            agent,
            Modification::ModifyPt {
                power: 5,
                toughness: 0,
            },
        );
        (s, ogre, agent)
    };
    let (mut s, _, _) = setup();
    let hand = s.zone_size(0, ZoneType::Hand);
    cast(&mut s, "Return of the Wildspeaker");
    choose_mode(&mut s, 0);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);

    let (mut s, ogre, agent) = setup();
    cast(&mut s, "Return of the Wildspeaker");
    choose_mode(&mut s, 1);
    settle(&mut s);
    let ogre_view = s.game().characteristics(ogre);
    assert_eq!((ogre_view.power, ogre_view.toughness), (5, 5));
    let agent_view = s.game().characteristics(agent);
    assert_eq!((agent_view.power, agent_view.toughness), (7, 3));
}
