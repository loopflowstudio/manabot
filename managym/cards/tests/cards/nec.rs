//! Release to Memory: one graveyard exile and its plain Spirit tokens.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, plant_in_graveyard},
    inspect::offered_targets,
    play::{attack_with, cast, pass_until, settle_targeting_opponent},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

#[test]
fn release_to_memory_exiles_only_the_opponents_graveyard_and_counts_creatures() {
    let mut s = five_color_setup(&["Release to Memory", "Gray Ogre", "Llanowar Elves"]);
    let own = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let ogre = plant_in_graveyard(&mut s, 1, "Gray Ogre");
    let elf = plant_in_graveyard(&mut s, 1, "Llanowar Elves");
    let land = plant_in_graveyard(&mut s, 1, "Forest");
    cast(&mut s, "Release to Memory");
    if s.action_space().kind == ActionSpaceKind::ChooseTarget {
        assert_eq!(offered_targets(&s), vec![Target::Player(PlayerId(1))]);
    }
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.zones.zone_of(own), Some(ZoneType::Graveyard));
    for card in [ogre, elf, land] {
        assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    }
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 0);
    let spirits = s.battlefield_permanents_named(0, "Spirit");
    assert_eq!(spirits.len(), 2);
    assert!(s.battlefield_permanents_named(1, "Spirit").is_empty());
    for id in spirits {
        let view = s.game().characteristics(id);
        assert_eq!(view.pt(), (1, 1));
        assert!(view.keywords.iter().next().is_none());
    }
}

#[test]
fn release_to_memory_counts_the_graveyard_at_resolution_and_not_preexisting_exile() {
    let mut s = five_color_setup(&["Release to Memory", "Gray Ogre", "Llanowar Elves"]);
    let ogre = plant_in_graveyard(&mut s, 1, "Gray Ogre");
    cast(&mut s, "Release to Memory");
    // A response removes one creature and puts a different one in the graveyard.
    s.game_mut().move_card(ogre, ZoneType::Exile);
    let elf = plant_in_graveyard(&mut s, 1, "Llanowar Elves");
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.zones.zone_of(elf), Some(ZoneType::Exile));
    assert_eq!(s.battlefield_permanents_named(0, "Spirit").len(), 1);
}

#[test]
fn release_to_memory_accepts_an_empty_or_noncreature_graveyard_without_tokens() {
    for noncreature in [false, true] {
        let mut s = five_color_setup(&["Release to Memory"]);
        if noncreature {
            plant_in_graveyard(&mut s, 1, "Forest");
        }
        cast(&mut s, "Release to Memory");
        settle_targeting_opponent(&mut s);
        assert_eq!(s.zone_size(1, ZoneType::Graveyard), 0);
        assert!(s.battlefield_permanents_named(0, "Spirit").is_empty());
    }
}

#[test]
fn release_to_memory_spirits_block_and_are_blocked_by_nonspirits() {
    for spirit_attacking in [false, true] {
        let mut s = five_color_setup(&["Gray Ogre"]);
        let spirit = s
            .game_mut()
            .create_token(
                "token.plain_spirit",
                PlayerId(usize::from(!spirit_attacking)),
                false,
            )
            .unwrap();
        let ogre = s.force_permanent_on_battlefield(usize::from(spirit_attacking), "Gray Ogre");
        let (attacker, blocker) = if spirit_attacking {
            (spirit, ogre)
        } else {
            (ogre, spirit)
        };
        clear_summoning_sickness(&mut s, attacker);
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, attacker);
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        assert!(s.action_space().actions.iter().any(|a| matches!(a,
            Action::DeclareBlocker { blocker: b, attacker: Some(a), .. } if *b == blocker && *a == attacker)));
    }
}
