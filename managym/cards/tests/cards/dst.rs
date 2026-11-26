//! Darksteel.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_to_hand, pin_top, until_end_of_turn},
    inspect::{card_of, now, zone_of},
    play::{activate, cast, select_card, settle},
};
use managym::{
    flow::event::GameEvent,
    state::{
        card::{Keyword, Keywords},
        effect::Modification,
        game_object::PlayerId,
        zone::ZoneType,
    },
};

#[test]
fn milling_colossus_reveals_and_shuffles_without_a_zone_change() {
    let mut s = five_color_setup(&["Darksteel Colossus", "Dryad Militant", "Sparring Dummy"]);
    s.force_permanent_on_battlefield(1, "Dryad Militant");
    let dummy = s.force_permanent_on_battlefield(0, "Sparring Dummy");
    clear_summoning_sickness(&mut s, dummy);
    let colossus = pin_top(&mut s, &["Darksteel Colossus"])[0];
    let definition = s.game().state[colossus].definition_id;
    let known = &mut s.game_mut().state.players[0].known_library;
    known.own.insert(0, definition);
    known.public.insert(0, definition);
    let count = s.zone_size(0, ZoneType::Library);
    activate(&mut s, dummy, 0);
    let mut fork = s.clone();
    for game in [&mut s, &mut fork] {
        let cursor = game.game().state.events.len();
        settle(game);
        assert_eq!(zone_of(game, colossus), Some(ZoneType::Library));
        assert_ne!(now(game, colossus), colossus);
        assert!(game.game().state.live(colossus).is_none());
        assert_eq!(game.zone_size(0, ZoneType::Library), count);
        assert!(game.game().state.players[0].known_library.is_empty());
        let events: Vec<_> = game.game().state.events.iter().skip(cursor).collect();
        assert_eq!(events.iter().filter(|e| matches!(e, GameEvent::CardRevealed { owner: PlayerId(0), definition: d } if *d == definition)).count(), 1);
        assert!(!events.iter().any(
            |e| matches!(e, GameEvent::CardMoved { departed: Some(o), .. } if *o == colossus)
        ));
    }
    assert_eq!(
        s.game().state.deterministic_hash(),
        fork.game().state.deterministic_hash()
    );
}

#[test]
fn discarding_colossus_reveals_the_card_and_returns_it_to_its_library() {
    let mut s = five_color_setup(&["Darksteel Colossus", "Seize the Spoils"]);
    s.set_player_mana_pool(0, "2R");
    let card = move_to_hand(&mut s, "Darksteel Colossus");
    let definition = s.game().state[card].definition_id;
    cast(&mut s, "Seize the Spoils");
    select_card(&mut s, card);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Library));
    assert_ne!(now(&s, card), card);
    assert!(s
        .game()
        .state
        .events
        .iter()
        .any(|e| matches!(e, GameEvent::CardRevealed { definition: d, .. } if *d == definition)));
}

#[test]
fn losing_indestructible_allows_destruction_and_losing_all_abilities_allows_death() {
    let mut s = five_color_setup(&["Darksteel Colossus", "Fumigate"]);
    let id = s.force_permanent_on_battlefield(1, "Darksteel Colossus");
    let card = card_of(&s, id);
    for (loss, zone) in [
        (None, ZoneType::Battlefield),
        (
            Some(Modification::LoseKeywords(Keywords::of(&[
                Keyword::Indestructible,
            ]))),
            ZoneType::Library,
        ),
        (Some(Modification::LoseAllAbilities), ZoneType::Graveyard),
    ] {
        let mut branch = s.clone();
        if let Some(loss) = loss {
            until_end_of_turn(&mut branch, id, loss);
        }
        cast(&mut branch, "Fumigate");
        settle(&mut branch);
        assert_eq!(zone_of(&branch, card), Some(zone));
    }
}
