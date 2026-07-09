// statics.rs
// Continuous-effect queries: effective power/toughness, effective keywords,
// animation-aware type checks, and static-condition evaluation.
//
// Layer order (simplified CR 613.3/613.4):
//   1. base P/T — printed, or characteristic-defining (CR 613.3a layer 7a:
//      Suki "power = creatures you control", Dragonfly Swarm "power =
//      noncreature nonland cards in your graveyard"); earthbend animation
//      sets base 0/0 by construction (lands print no P/T).
//   2. static continuous P/T effects (layer 7c: anthems and conditional
//      self-buffs), applied while their source is on the battlefield.
//   3. until-end-of-turn P/T modifications (also layer 7c; all Milestone-1
//      effects are additive, so ordering within the layer is immaterial and
//      no timestamp system is kept).
//   4. +1/+1 counters (layer 7d).
// Anthem scopes match printed characteristics plus animation (never
// power), so the computation cannot recurse.

use crate::{
    flow::game::Game,
    state::{
        ability::StaticCondition,
        card::{Keywords, PowerCda, StaticScope},
        game_object::{PermanentId, PlayerId},
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

impl Game {
    /// The permanent's effective power: CDA-aware base, plus static buffs,
    /// until-EOT deltas, and +1/+1 counters.
    pub fn effective_power(&self, permanent_id: PermanentId) -> i32 {
        self.effective_pt(permanent_id).0
    }

    /// The permanent's effective toughness (same layering as power).
    pub fn effective_toughness(&self, permanent_id: PermanentId) -> i32 {
        self.effective_pt(permanent_id).1
    }

    pub fn effective_pt(&self, permanent_id: PermanentId) -> (i32, i32) {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return (0, 0);
        };
        let card = &self.state.cards[permanent.card];

        // Layer 7a: base P/T (characteristic-defining, or printed).
        let base_power = match &card.power_cda {
            Some(cda) => self.cda_power(cda, permanent.controller),
            None => card.power.unwrap_or(0),
        };
        let base_toughness = card.toughness.unwrap_or(0);

        // Layer 7c: static continuous effects from battlefield permanents.
        let (static_power, static_toughness) = self.static_pt_bonus(permanent_id);

        (
            base_power + static_power + permanent.temp_power + permanent.plus1_counters,
            base_toughness + static_toughness + permanent.temp_toughness + permanent.plus1_counters,
        )
    }

    fn cda_power(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CreaturesYouControl => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| self.permanent_is_creature(*id))
                .count() as i32,
            PowerCda::GraveyardMatching(predicate) => {
                self.count_graveyard_matching(controller, predicate) as i32
            }
        }
    }

    /// Sum of static P/T buffs applying to `target_id` from every
    /// battlefield permanent (including itself for `StaticScope::This`).
    fn static_pt_bonus(&self, target_id: PermanentId) -> (i32, i32) {
        let Some(target) = self.state.permanents[target_id].as_ref() else {
            return (0, 0);
        };
        let target_card = &self.state.cards[target.card];
        let mut power = 0;
        let mut toughness = 0;

        for player in [PlayerId(0), PlayerId(1)] {
            for source_card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                let card = &self.state.cards[source_card_id];
                if card.static_pt_buffs.is_empty() {
                    continue;
                }
                let Some(source_perm_id) = self.state.card_to_permanent[source_card_id] else {
                    continue;
                };
                let Some(source_perm) = self.state.permanents[source_perm_id].as_ref() else {
                    continue;
                };
                for buff in &card.static_pt_buffs {
                    let applies = match &buff.scope {
                        StaticScope::This => source_perm_id == target_id,
                        StaticScope::OtherYouControl(predicate) => {
                            source_perm_id != target_id
                                && source_perm.controller == target.controller
                                // Printed characteristics + animation only
                                // (no power predicates — keeps this
                                // non-recursive).
                                && predicate.matches(
                                    target_card,
                                    target_card.power.unwrap_or(0),
                                    target.animated,
                                )
                        }
                    };
                    if !applies {
                        continue;
                    }
                    if let Some(condition) = &buff.condition {
                        if !self.check_static_condition(condition, source_perm.controller) {
                            continue;
                        }
                    }
                    power += buff.power;
                    toughness += buff.toughness;
                }
            }
        }
        (power, toughness)
    }

    /// Evaluate a static condition for `controller` ("you").
    pub(crate) fn check_static_condition(
        &self,
        condition: &StaticCondition,
        controller: PlayerId,
    ) -> bool {
        match condition {
            StaticCondition::GraveyardAtLeast { count, predicate } => {
                self.count_graveyard_matching(controller, predicate) >= *count
            }
        }
    }

    /// Animation-aware creature check.
    pub(crate) fn permanent_is_creature(&self, permanent_id: PermanentId) -> bool {
        self.state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
            .is_some_and(|permanent| permanent.is_creature(&self.state.cards[permanent.card]))
    }

    /// Printed keywords plus until-EOT grants plus animation haste.
    pub(crate) fn effective_keywords(&self, permanent_id: PermanentId) -> Keywords {
        self.state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
            .map(|permanent| permanent.effective_keywords(&self.state.cards[permanent.card]))
            .unwrap_or_default()
    }

    /// Lethal-damage check (CR 704.5g), animation- and statics-aware.
    pub(crate) fn has_lethal_damage(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if !self.permanent_is_creature(permanent_id) {
            return false;
        }
        if permanent.deathtouch_damage && permanent.damage > 0 {
            return true;
        }
        permanent.damage >= self.effective_toughness(permanent_id)
    }

    /// Match a battlefield permanent against a predicate using its
    /// effective power and animation-aware types.
    pub(crate) fn permanent_matches_predicate(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let card = &self.state.cards[permanent.card];
        predicate.matches(card, self.effective_power(permanent_id), permanent.animated)
    }
}
