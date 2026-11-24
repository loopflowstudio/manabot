//! Card behavior tests, one module per set module in `src/`: a test lives
//! with the set that first printed the card it is about. Engine rules that
//! are not about one card stay in the engine's `rules_tests`. Tests never
//! decide whether a card is supported; see `cards/README.md` for coverage.
//! The scenario helpers are the engine's `tests/support`, mounted by path so
//! each primitive stays defined once.

#[path = "../../tests/support/mod.rs"]
mod support;

#[path = "cards/aer.rs"]
mod aer;
#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/akh.rs"]
mod akh;
#[path = "cards/atq.rs"]
mod atq;
#[path = "cards/avr.rs"]
mod avr;
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
#[path = "cards/dka.rs"]
mod dka;
#[path = "cards/dmu.rs"]
mod dmu;
#[path = "cards/dom.rs"]
mod dom;
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
#[path = "cards/frf.rs"]
mod frf;
#[path = "cards/fut.rs"]
mod fut;
#[path = "cards/gpt.rs"]
mod gpt;
#[path = "cards/grn.rs"]
mod grn;
#[path = "cards/gtc.rs"]
mod gtc;
#[path = "cards/hob.rs"]
mod hob;
#[path = "cards/hou.rs"]
mod hou;
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
#[path = "cards/jmp.rs"]
mod jmp;
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
#[path = "cards/lrw.rs"]
mod lrw;
#[path = "cards/m10.rs"]
mod m10;
#[path = "cards/m12.rs"]
mod m12;
#[path = "cards/m14.rs"]
mod m14;
#[path = "cards/m15.rs"]
mod m15;
#[path = "cards/m19.rs"]
mod m19;
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
#[path = "cards/mor.rs"]
mod mor;
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
#[path = "cards/ogw.rs"]
mod ogw;
#[path = "cards/ons.rs"]
mod ons;
#[path = "cards/ori.rs"]
mod ori;
#[path = "cards/p02.rs"]
mod p02;
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
#[path = "cards/tmp.rs"]
mod tmp;
#[path = "cards/tsp.rs"]
mod tsp;
#[path = "cards/ulg.rs"]
mod ulg;
#[path = "cards/usg.rs"]
mod usg;
#[path = "cards/vis.rs"]
mod vis;
#[path = "cards/war.rs"]
mod war;
#[path = "cards/woe.rs"]
mod woe;
#[path = "cards/wth.rs"]
mod wth;
#[path = "cards/xln.rs"]
mod xln;
#[path = "cards/zen.rs"]
mod zen;
#[path = "cards/znr.rs"]
mod znr;

/// Play an unblocked combat with just the printed creature on the battlefield.
fn printed_creature_combat(
    name: &str,
    power: i32,
    toughness: i32,
    keywords: &[managym::state::card::Keyword],
) {
    use managym::{flow::turn::StepKind, state::card::Keyword};
    use support::{
        game::five_color_setup,
        inspect::permanent_named,
        play::{attack_with, play_and_settle},
    };
    let mut s = five_color_setup(&[name]);
    s.set_player_mana_pool(0, "20WWWWWWUUUUUUBBBBBBRRRRRRGGGGGG");
    play_and_settle(&mut s, name);
    let id = permanent_named(&s, name);
    let face = s.game().characteristics(id);
    assert_eq!(face.pt(), (power, toughness));
    let keywords = managym::state::card::Keywords::of(keywords);
    assert_eq!(face.keywords, keywords);
    if !keywords.has(Keyword::Haste) {
        s.advance_to_active_step(1, StepKind::Main);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, id);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    let damage = power
        * if keywords.has(Keyword::DoubleStrike) {
            2
        } else {
            1
        };
    s.assert_life(1, 20 - damage);
    s.assert_life(
        0,
        20 + if keywords.has(Keyword::Lifelink) {
            damage
        } else {
            0
        },
    );
    assert_eq!(
        support::inspect::tapped(&s, id),
        !keywords.has(Keyword::Vigilance)
    );
}

/// Check the controller's tokens in the resolved position, including any active team bonus.
fn assert_creature_tokens(
    s: &support::game::Scenario,
    name: &str,
    count: usize,
    pt: (i32, i32),
    colors: &[managym::state::mana::Color],
    keywords: &[managym::state::card::Keyword],
) {
    use managym::state::{card::Keywords, game_object::ObjectKind, mana::Colors};
    let tokens = s.battlefield_permanents_named(0, name);
    assert_eq!(tokens.len(), count);
    let subtypes: Vec<_> = name.split_whitespace().collect();
    for id in tokens {
        assert_eq!(
            s.game().state[support::inspect::card_of(s, id)].kind(),
            ObjectKind::Token
        );
        let face = s.game().characteristics(id);
        assert_eq!(face.pt(), pt);
        assert_eq!(face.creature_types, subtypes);
        assert_eq!(face.colors, Colors::from_iter(colors.iter().copied()));
        assert_eq!(face.keywords, Keywords::of(keywords));
    }
}
