// statics.rs
// Continuous-effect queries: effective power/toughness, effective keywords,
// animation-aware type checks, and static-condition evaluation.
//
// Supported subset of CR 613.4 (September 25, 2026):
//   1. base P/T — printed, or characteristic-defining (CR 613.4a layer 7a:
//      Suki "power = creatures you control", Dragonfly Swarm "power =
//      noncreature nonland cards in your graveyard"); earthbend animation
//      sets base 0/0 by construction (lands print no P/T).
//   2. until-end-of-turn effects that set base power or toughness (layer
//      7b: Flexible Waterbender, Turtle-Duck). Each replaces the 7a value;
//      a later setter overwrites an earlier one, which is timestamp order.
//   3. static continuous P/T effects (layer 7c: anthems and conditional
//      self-buffs), applied while their source is on the battlefield.
//   4. until-end-of-turn P/T modifications (also layer 7c; all Milestone-1
//      effects are additive, so ordering within the layer is immaterial and
//      no timestamp system is kept).
//   5. +1/+1 counters (also layer 7c, CR 613.4c).
// These additive layer-7c terms commute. Layer 7d switches P/T, and is not
// represented here; neither are general timestamp/dependency interactions.
// Static keyword grants (layer 6) add to printed and until-EOT keywords.
// "Equipped/enchanted creature" effects follow the exact attached object.
// Static restrictions (can't attack or block, doesn't untap) use the same
// scopes.
// Scopes match printed characteristics, animation, counters and temporary
// subtypes (never power). A P/T scope's flying test reads effective
// keywords; a keyword-grant scope's reads only non-static keywords. Neither
// computation can recurse.

use crate::{
    flow::game::Game,
    state::{
        ability::StaticCondition,
        card::{Card, Keywords, PowerCda, Restriction, StaticScope},
        game_object::{PermanentId, PlayerId},
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// Subtypes that permanents in the current pools can carry which are not
/// creature types (land and artifact types; CR 205.3).
const NONCREATURE_SUBTYPES: [&str; 10] = [
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Clue",
    "Food",
    "Treasure",
    "Equipment",
    "Vehicle",
];

impl Game {
    /// The permanent's effective power: CDA-aware or set base, plus static buffs,
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
        // Layer 7b: until-EOT base setters replace the 7a value.
        let base_power = permanent.temp_base_power.unwrap_or(base_power);
        let base_toughness = permanent.temp_base_toughness.unwrap_or(base_toughness);

        // Layer 7c: static continuous effects from battlefield permanents.
        let (static_power, static_toughness) = self.static_pt_bonus(permanent_id);

        (
            base_power + static_power + permanent.temp_power + permanent.plus1_counters,
            base_toughness + static_toughness + permanent.temp_toughness + permanent.plus1_counters,
        )
    }

    fn cda_power(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter_map(|id| self.state.permanents[id].as_ref())
                .filter(|p| self.state.cards[p.card].types.is_land())
                .map(|p| p.plus1_counters)
                .sum(),
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
        let mut power = 0;
        let mut toughness = 0;
        for (source_id, controller, card) in self.static_sources() {
            for buff in &card.static_pt_buffs {
                if !self.static_scope_applies(&buff.scope, source_id, controller, target_id, true)
                    || buff.condition.as_ref().is_some_and(|condition| {
                        !self.check_static_condition(condition, controller)
                    })
                {
                    continue;
                }
                let times = buff
                    .per_creature_type_max
                    .map_or(1, |max| self.creature_type_count(target_id).min(max));
                power += buff.power * times;
                toughness += buff.toughness * times;
            }
        }
        (power, toughness)
    }

    /// Keywords granted to `target_id` by static abilities of battlefield
    /// permanents (layer 6).
    fn static_keyword_bonus(&self, target_id: PermanentId) -> Keywords {
        let mut keywords = Keywords::default();
        for (source_id, controller, card) in self.static_sources() {
            for grant in &card.static_keyword_grants {
                if self.static_scope_applies(&grant.scope, source_id, controller, target_id, false)
                    && grant
                        .condition
                        .as_ref()
                        .is_none_or(|condition| self.check_static_condition(condition, controller))
                {
                    keywords = keywords.union(&grant.keywords);
                }
            }
        }
        keywords
    }

    /// Battlefield permanents whose cards project static P/T, keyword or
    /// restriction effects, with their controllers.
    fn static_sources(&self) -> impl Iterator<Item = (PermanentId, PlayerId, &Card)> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(move |player| {
                self.state
                    .zones
                    .zone_cards(ZoneType::Battlefield, player)
                    .iter()
                    .filter_map(move |&card_id| {
                        let card = &self.state.cards[card_id];
                        if card.static_pt_buffs.is_empty()
                            && card.static_keyword_grants.is_empty()
                            && card.static_restrictions.is_empty()
                        {
                            return None;
                        }
                        let permanent_id = self.state.card_to_permanent[card_id]?;
                        let permanent = self.state.permanents[permanent_id].as_ref()?;
                        Some((permanent_id, permanent.controller, card))
                    })
            })
    }

    /// Whether a static scope from `source_id` (controlled by `controller`)
    /// covers `target_id`. `effective_flying` selects the P/T layer's view of
    /// flying; keyword grants read only non-static keywords.
    fn static_scope_applies(
        &self,
        scope: &StaticScope,
        source_id: PermanentId,
        controller: PlayerId,
        target_id: PermanentId,
        effective_flying: bool,
    ) -> bool {
        let Some(target) = self.state.permanents[target_id].as_ref() else {
            return false;
        };
        let (predicate, other, with_counters) = match scope {
            StaticScope::This => return source_id == target_id,
            StaticScope::Attached => return self.attached_creature(source_id) == Some(target_id),
            StaticScope::OtherYouControl(predicate) => (predicate, true, false),
            StaticScope::YouControl(predicate) => (predicate, false, false),
            StaticScope::YouControlWithCounters(predicate) => (predicate, false, true),
        };
        if target.controller != controller
            || (other && source_id == target_id)
            || (with_counters && !target.has_counters())
        {
            return false;
        }
        // Printed power only (no power predicates), keeping P/T non-recursive.
        let card = &self.state.cards[target.card];
        self.permanent_matches_with(target_id, predicate, card.power.unwrap_or(0), || {
            if effective_flying {
                self.effective_keywords(target_id).flying
            } else {
                target.effective_keywords(card).flying
            }
        })
    }

    /// The permanent an Aura or Equipment is attached to, if that exact
    /// object is still on the battlefield.
    pub(crate) fn attached_permanent(&self, source: PermanentId) -> Option<PermanentId> {
        let attached = self.state.permanents[source].as_ref()?.attached_to?;
        self.lookup_current_permanent(attached).ok()
    }

    /// The attached permanent, if it is still a creature.
    pub(crate) fn attached_creature(&self, source: PermanentId) -> Option<PermanentId> {
        self.attached_permanent(source)
            .filter(|&creature| self.permanent_is_creature(creature))
    }

    /// Whether a static restriction from a battlefield permanent applies to
    /// `target_id`.
    pub(crate) fn has_restriction(&self, target_id: PermanentId, kind: Restriction) -> bool {
        self.static_sources().any(|(source_id, controller, card)| {
            card.static_restrictions.iter().any(|restriction| {
                restriction.kind == kind
                    && self.static_scope_applies(
                        &restriction.scope,
                        source_id,
                        controller,
                        target_id,
                        false,
                    )
            })
        })
    }

    /// Distinct creature types of a battlefield creature (CR 205.3m): printed
    /// subtypes of a printed creature, less noncreature subtypes, plus
    /// temporarily granted subtypes. Animated lands keep only granted types.
    fn creature_type_count(&self, permanent_id: PermanentId) -> i32 {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return 0;
        };
        let card = &self.state.cards[permanent.card];
        let mut types: Vec<&str> = Vec::new();
        if card.types.is_creature() {
            types.extend(
                card.subtypes
                    .iter()
                    .map(String::as_str)
                    .filter(|s| !NONCREATURE_SUBTYPES.contains(s)),
            );
        }
        types.extend(permanent.temp_subtypes.iter().map(String::as_str));
        types.sort_unstable();
        types.dedup();
        types.len() as i32
    }

    /// Evaluate a static condition for `controller` ("you").
    pub(crate) fn check_static_condition(
        &self,
        condition: &StaticCondition,
        controller: PlayerId,
    ) -> bool {
        match condition {
            StaticCondition::Always => true,
            StaticCondition::YourTurn => self.active_player() == controller,
            StaticCondition::CardsDrawnAtLeast { count } => {
                self.state.turn.cards_drawn_this_turn[controller.0] >= *count
            }
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

    /// Printed keywords plus until-EOT grants plus animation haste plus
    /// static grants from battlefield permanents.
    pub(crate) fn effective_keywords(&self, permanent_id: PermanentId) -> Keywords {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return Keywords::default();
        };
        permanent
            .effective_keywords(&self.state.cards[permanent.card])
            .union(&self.static_keyword_bonus(permanent_id))
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
    /// effective power, keywords and animation-aware types.
    pub(crate) fn permanent_matches_predicate(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        self.permanent_matches_with(
            permanent_id,
            predicate,
            self.effective_power(permanent_id),
            || self.effective_keywords(permanent_id).flying,
        )
    }

    /// Match a battlefield permanent against a predicate, counting its
    /// temporary subtypes and animation. The caller supplies the power and
    /// flying views it can read without recursing.
    fn permanent_matches_with(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
        power: i32,
        flying: impl FnOnce() -> bool,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if (predicate.flying && !flying()) || (predicate.no_counters && permanent.has_counters()) {
            return false;
        }
        let card = &self.state.cards[permanent.card];
        let has_subtype = |name: &str| {
            card.has_subtype(name) || permanent.temp_subtypes.iter().any(|s| s == name)
        };
        if predicate
            .subtype
            .as_deref()
            .is_some_and(|s| !has_subtype(s))
            || (!predicate.subtypes_any.is_empty()
                && !predicate.subtypes_any.iter().any(|s| has_subtype(s)))
            || predicate.not_subtypes.iter().any(|s| has_subtype(s))
        {
            return false;
        }
        let structural = CardPredicate {
            flying: false,
            subtype: None,
            subtypes_any: Vec::new(),
            not_subtypes: Vec::new(),
            ..predicate.clone()
        };
        structural.matches(card, power, permanent.animated)
    }
}
