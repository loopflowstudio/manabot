//! Card behavior tests, one module per set module in `src/`: a test lives
//! with the set that first printed the card it is about. Engine rules that
//! are not about one card stay in the engine's `rules_tests`. Tests never
//! decide whether a card is supported; see `cards/README.md` for coverage.
//! The scenario helpers are the engine's `tests/support`, mounted by path so
//! each primitive stays defined once.

#[path = "../../tests/support/mod.rs"]
mod support;

use managym::agent::env::GameSnapshot;
use support::game::{deck, Scenario};

// Constructed card positions use the benchmark fixture access enabled by this crate.
impl Scenario {
    /// Capture this constructed decision for exact native forks.
    pub fn snapshot(&self) -> GameSnapshot {
        use managym::{state::player::PlayerConfig, Env};
        let mut env = Env::new(1, false, false, false);
        env.reset(
            ["a", "b"]
                .map(|name| PlayerConfig::new(name, deck(&[("Island", 40)])))
                .to_vec(),
        )
        .unwrap();
        *env.benchmark_game_mut() = self.game().clone();
        env.snapshot().unwrap()
    }

    fn restore(&mut self, snapshot: &GameSnapshot) {
        let mut env = snapshot.restore(Default::default()).unwrap();
        *self.game_mut() = env.benchmark_game_mut().clone();
    }
}

#[path = "cards/aer.rs"]
mod aer;
#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/atq.rs"]
mod atq;
#[path = "cards/bfz.rs"]
mod bfz;
#[path = "cards/blb.rs"]
mod blb;
#[path = "cards/bng.rs"]
mod bng;
#[path = "cards/bro.rs"]
mod bro;
#[path = "cards/c17.rs"]
mod c17;
#[path = "cards/c19.rs"]
mod c19;
#[path = "cards/chk.rs"]
mod chk;
#[path = "cards/clb.rs"]
mod clb;
#[path = "cards/cmr.rs"]
mod cmr;
#[path = "cards/csp.rs"]
mod csp;
#[path = "cards/definitions.rs"]
mod definitions;
#[path = "cards/dft.rs"]
mod dft;
#[path = "cards/dgm.rs"]
mod dgm;
#[path = "cards/dmu.rs"]
mod dmu;
#[path = "cards/dom.rs"]
mod dom;
#[path = "cards/dsk.rs"]
mod dsk;
#[path = "cards/dtk.rs"]
mod dtk;
#[path = "cards/ecl.rs"]
mod ecl;
#[path = "cards/eld.rs"]
mod eld;
#[path = "cards/emn.rs"]
mod emn;
#[path = "cards/eoe.rs"]
mod eoe;
#[path = "cards/fdn.rs"]
mod fdn;
#[path = "cards/fin.rs"]
mod fin;
#[path = "cards/frf.rs"]
mod frf;
#[path = "cards/fut.rs"]
mod fut;
#[path = "cards/gpt.rs"]
mod gpt;
#[path = "cards/gtc.rs"]
mod gtc;
#[path = "cards/hob.rs"]
mod hob;
#[path = "cards/ice.rs"]
mod ice;
#[path = "cards/iko.rs"]
mod iko;
#[path = "cards/inv.rs"]
mod inv;
#[path = "cards/isd.rs"]
mod isd;
#[path = "cards/j22.rs"]
mod j22;
#[path = "cards/jou.rs"]
mod jou;
#[path = "cards/khm.rs"]
mod khm;
#[path = "cards/kld.rs"]
mod kld;
#[path = "cards/ktk.rs"]
mod ktk;
#[path = "cards/lea.rs"]
mod lea;
#[path = "cards/m12.rs"]
mod m12;
#[path = "cards/m14.rs"]
mod m14;
#[path = "cards/m15.rs"]
mod m15;
#[path = "cards/m20.rs"]
mod m20;
#[path = "cards/mbs.rs"]
mod mbs;
#[path = "cards/mh1.rs"]
mod mh1;
#[path = "cards/mid.rs"]
mod mid;
#[path = "cards/mkm.rs"]
mod mkm;
#[path = "cards/mmq.rs"]
mod mmq;
#[path = "cards/mrd.rs"]
mod mrd;
#[path = "cards/msh.rs"]
mod msh;
#[path = "cards/nec.rs"]
mod nec;
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/ody.rs"]
mod ody;
#[path = "cards/ons.rs"]
mod ons;
#[path = "cards/ori.rs"]
mod ori;
#[path = "cards/pls.rs"]
mod pls;
#[path = "cards/por.rs"]
mod por;
#[path = "cards/psal.rs"]
mod psal;
#[path = "cards/ptk.rs"]
mod ptk;
#[path = "cards/rav.rs"]
mod rav;
#[path = "cards/rix.rs"]
mod rix;
#[path = "cards/roe.rs"]
mod roe;
#[path = "cards/rtr.rs"]
mod rtr;
#[path = "cards/scryfall.rs"]
mod scryfall;
#[path = "cards/shm.rs"]
mod shm;
#[path = "cards/soi.rs"]
mod soi;
#[path = "cards/spm.rs"]
mod spm;
#[path = "cards/sth.rs"]
mod sth;
#[path = "cards/stx.rs"]
mod stx;
#[path = "cards/tdm.rs"]
mod tdm;
#[path = "cards/ths.rs"]
mod ths;
#[path = "cards/tla.rs"]
mod tla;
#[path = "cards/tle.rs"]
mod tle;
#[path = "cards/ulg.rs"]
mod ulg;
#[path = "cards/vis.rs"]
mod vis;
#[path = "cards/war.rs"]
mod war;
#[path = "cards/woe.rs"]
mod woe;
#[path = "cards/wth.rs"]
mod wth;
#[path = "cards/zen.rs"]
mod zen;
#[path = "cards/znr.rs"]
mod znr;

/// Both Clerics share a graveyard and a competing restricted spend. Only
/// their second legal target and their kicker price differ.
fn check_kicked_return(name: &str, mana: &str, second_target: &str, excluded: &str) {
    use managym::{
        agent::{
            action::{Action, ActionSpaceKind},
            observation::Observation,
        },
        state::{
            game_object::{PlayerId, Target},
            mana::{Color, Mana, ManaRestriction},
            zone::ZoneType,
        },
    };
    use support::{
        game::{five_color_unfunded_setup, move_card, move_from_library},
        inspect::{offered_targets, permanent_of, top_spell, zone_of},
        play::{
            cast, cast_card, command, pass_until, resolve_top, settle, spend_colorless, spend_named,
        },
    };

    let mut s = five_color_unfunded_setup(&[
        name,
        "Llanowar Elves",
        "Sol Ring",
        "Gray Ogre",
        "Lightning Bolt",
    ]);
    let elves = move_from_library(&mut s, 0, "Llanowar Elves", ZoneType::Graveyard);
    let other = move_from_library(&mut s, 0, second_target, ZoneType::Graveyard);
    for excluded in ["Forest", "Lightning Bolt", excluded] {
        move_from_library(&mut s, 0, excluded, ZoneType::Graveyard);
    }
    move_from_library(&mut s, 1, "Llanowar Elves", ZoneType::Graveyard);
    let bolt = move_from_library(&mut s, 1, "Lightning Bolt", ZoneType::Hand);
    s.set_player_mana_pool(1, "R");
    s.set_player_mana_pool(0, mana);
    s.game_mut().state.players[0].mana_pool.add(
        &Mana::restricted_to(ManaRestriction::spell(["Cleric"]), [0, 0, 0, 0, 1, 0]),
        None,
    );

    let card = cast(&mut s, name);
    let kicker = s.snapshot();
    command(&mut s, |a| matches!(a, Action::Decline { .. }));
    spend_colorless(&mut s);
    assert_eq!(top_spell(&s).spent.total(), 2);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack_objects.is_empty());

    s.restore(&kicker);
    command(&mut s, |a| matches!(a, Action::PayCost { .. }));
    spend_named(&mut s, |label| label.contains("{G}"));
    spend_colorless(&mut s);
    let spent = top_spell(&s).spent.clone();
    assert_eq!(spent.total(), Mana::parse(mana).total());
    assert_eq!(spent.amounts()[Color::Green as usize], 1);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let targets = offered_targets(&s);
    assert_eq!(targets.len(), 2);
    let source = permanent_of(&s, card).unwrap();
    let entry = s.game().state.permanents[source]
        .as_ref()
        .unwrap()
        .spell_entry
        .as_deref()
        .unwrap();
    assert!(entry.kicked);
    assert_eq!(entry.payment.as_ref(), Some(&spent));
    let chooser = Observation::for_player(s.game(), PlayerId(0));
    assert!(chooser.continuation.iter().any(|frame| frame.kicked));
    let targeting = s.snapshot();
    for (chosen, stays) in [(elves, other), (other, elves)] {
        s.restore(&targeting);
        let target = Target::GraveyardCard(chosen);
        assert!(targets.contains(&target));
        command(
            &mut s,
            |a| matches!(a, Action::ChooseTarget { target: t, .. } if *t == target),
        );
        s.pass_priority();
        cast_card(&mut s, bolt);
        assert!(s.choose_target(Target::Permanent(source)));
        resolve_top(&mut s);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&s, chosen), Some(ZoneType::Graveyard));
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            assert_eq!(obs.stack_objects.len(), 1);
            assert!(obs.stack_objects[0].kicked);
            assert!(obs.continuation[0].kicked);
        }
        settle(&mut s);
        assert_eq!(zone_of(&s, chosen), Some(ZoneType::Battlefield));
        assert_eq!(zone_of(&s, stays), Some(ZoneType::Graveyard));
    }

    // Blink the kicked incarnation while its original trigger waits.
    s.restore(&targeting);
    command(
        &mut s,
        |a| matches!(a, Action::ChooseTarget {target: Target::GraveyardCard(id), ..} if *id == elves),
    );
    let exile = move_card(&mut s, card, ZoneType::Exile);
    let returned = move_card(&mut s, exile, ZoneType::Battlefield);
    let id = permanent_of(&s, returned).unwrap();
    assert!(s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .spell_entry
        .is_none());
    assert!(s.game().state.pending_triggers.is_empty());
    settle(&mut s);
    assert_eq!(zone_of(&s, elves), Some(ZoneType::Battlefield));
    assert_eq!(zone_of(&s, other), Some(ZoneType::Graveyard));
}

/// The objects in both graveyards after a mass return resolves, followed to
/// their current appearances.
struct GraveyardReturn {
    momo: managym::state::game_object::ObjectId,
    sailor: managym::state::game_object::ObjectId,
    drake: managym::state::game_object::ObjectId,
    bolt: managym::state::game_object::ObjectId,
    their_sailor: managym::state::game_object::ObjectId,
}

/// Cast `spell` over two graveyards: the caster's Momo, Spectral Sailor
/// (mana value 1, flying), Wind Drake (mana value 3, flying) and Lightning
/// Bolt, and the opponent's Spectral Sailor. Momo triggers once for each
/// other flier entering under its controller's control together with it.
fn return_from_two_graveyards(spell: &str) -> (support::game::Scenario, GraveyardReturn) {
    use support::{
        game::{five_color_setup, plant_in_graveyard},
        inspect::now,
        play::{cast, settle, spend_colorless},
    };
    let mut s = five_color_setup(&[
        spell,
        "Momo, Friendly Flier",
        "Spectral Sailor",
        "Wind Drake",
        "Lightning Bolt",
    ]);
    let momo = plant_in_graveyard(&mut s, 0, "Momo, Friendly Flier");
    let sailor = plant_in_graveyard(&mut s, 0, "Spectral Sailor");
    let drake = plant_in_graveyard(&mut s, 0, "Wind Drake");
    let bolt = plant_in_graveyard(&mut s, 0, "Lightning Bolt");
    let their_sailor = plant_in_graveyard(&mut s, 1, "Spectral Sailor");
    cast(&mut s, spell);
    spend_colorless(&mut s);
    settle(&mut s);
    let moved = GraveyardReturn {
        momo: now(&s, momo),
        sailor: now(&s, sailor),
        drake: now(&s, drake),
        bolt: now(&s, bolt),
        their_sailor: now(&s, their_sailor),
    };
    (s, moved)
}
