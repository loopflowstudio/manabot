//! Tests for cards filed in `src/mmq.rs`, in card order.

use managym::{
    agent::{action::ActionSpaceKind, observation::Observation},
    state::{game_object::PlayerId, zone::ZoneType},
};

use crate::support::{
    game::five_color_setup,
    inspect::{card_of, library_card},
    play::{cast, resolve_top, select_card, settle, target_opponent},
};

/// Bribery shows the searcher the opponent's library, not their hand, and
/// puts the chosen creature onto the battlefield under the caster's control.
#[test]
fn bribery_puts_a_creature_from_the_opponents_library_under_your_control() {
    let mut s = five_color_setup(&["Bribery", "Gray Ogre"]);
    let held = library_card(&s, 1, "Island");
    s.game_mut().move_card(held, ZoneType::Hand);
    let ogre = library_card(&s, 1, "Gray Ogre");
    cast(&mut s, "Bribery");
    target_opponent(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);

    let library = s.game().state.zones.size(ZoneType::Library, PlayerId(1));
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let shown = |zone: ZoneType| {
        observation
            .opponent_cards
            .iter()
            .filter(|card| card.zone == zone)
            .count()
    };
    assert_eq!(shown(ZoneType::Library), library);
    assert_eq!(shown(ZoneType::Hand), 0, "the hand stays hidden");

    // The searched library is exact for the searcher, so a sampled world
    // keeps it (and the hidden hand it implies) in place.
    let zones = |game: &managym::flow::game::Game| {
        [ZoneType::Hand, ZoneType::Library]
            .map(|zone| game.state.zones.zone_cards(zone, PlayerId(1)).to_vec())
    };
    let mut sampled = s.game().clone();
    sampled.determinize(PlayerId(0), 7).unwrap();
    assert_eq!(zones(&sampled), zones(s.game()));

    select_card(&mut s, ogre);
    settle(&mut s);
    let taken = s.game().state.card_to_permanent[ogre].expect("entered");
    let permanent = s.game().state.permanents[taken].as_ref().unwrap();
    assert_eq!(permanent.controller, PlayerId(0));
    assert_eq!(s.game().state.cards[card_of(&s, taken)].owner, PlayerId(1));
    assert_eq!(
        s.game().state.zones.zone_of(held),
        Some(ZoneType::Hand),
        "only the library is searched"
    );
}
