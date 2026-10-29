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
// Static type grants (layer 4) add card types; static keyword grants
// (layer 6) add to printed and until-EOT keywords. Earthbend animation sets
// base 0/0 (CR 701.66a) even on a permanent that prints P/T.
// "Equipped/enchanted creature" effects follow the exact attached object.
// An Aura that sets characteristics (`SetCharacteristics`) replaces the
// enchanted creature's name and creature types, sets base P/T in layer 7b
// before until-end-of-turn setters, and removes its abilities; timestamps
// are not kept, so every other ability and grant goes with them.
// Static restrictions (can't attack or block, doesn't untap, blocker limits)
// use the same scopes.
// A scope reads what the layers before its own determine (`ScopeView`):
// type grants read printed types; keyword grants add granted types; P/T
// scopes add effective keywords; restrictions, which no layer reads, also
// read effective power. No computation can recurse.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    flow::game::Game,
    state::{
        ability::StaticCondition,
        card::{Card, CardType, Keywords, PowerCda, Restriction, SetCharacteristics, StaticScope},
        game_object::{CardId, PermanentId, PlayerId},
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

/// Which characteristics a static scope's predicate reads (see the header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScopeView {
    /// Layer 4: printed types, printed power, non-static keywords.
    Types,
    /// Layer 6: adds granted types.
    Keywords,
    /// Layer 7: adds effective keywords.
    PowerToughness,
    /// Rules effects and ordinary predicates: adds effective power.
    Rules,
}

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

        // Layer 7a: base P/T (characteristic-defining, or printed); an
        // earthbent permanent is a 0/0 creature. A characteristic-setting
        // Aura's 7b base applies before until-EOT setters.
        let (base_power, base_toughness) = if let Some(set) = self.set_characteristics(permanent_id)
        {
            (set.power, set.toughness)
        } else if permanent.animated {
            (0, 0)
        } else {
            let power = match &card.power_cda {
                Some(cda) => self.cda_power(cda, permanent.controller),
                None => card.power.unwrap_or(0),
            };
            (power, card.toughness.unwrap_or(0))
        };
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
                .filter(|&id| self.permanent_is_land(id))
                .filter_map(|id| self.state.permanents[id].as_ref())
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
                if !self.static_scope_applies(
                    &buff.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::PowerToughness,
                ) || buff
                    .condition
                    .as_ref()
                    .is_some_and(|condition| !self.check_static_condition(condition, controller))
                {
                    continue;
                }
                let per_type = buff.per_creature_type_max.map_or(1, |max| {
                    (self.creature_types(target_id).len() as i32).min(max)
                });
                let per_color = buff
                    .per_color_among
                    .as_ref()
                    .map_or(1, |predicate| self.colors_among(controller, predicate));
                let times = per_type * per_color;
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
                if self.static_scope_applies(
                    &grant.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::Keywords,
                ) && grant
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

    /// A stack spell's keywords: printed, plus static grants to spells its
    /// controller controls (CR 613.1f).
    pub(crate) fn spell_keywords(&self, card_id: CardId, controller: PlayerId) -> Keywords {
        let card = &self.state.cards[card_id];
        let mut keywords = card.keywords;
        for (_, source_controller, source) in self.static_sources() {
            for grant in &source.static_keyword_grants {
                if let StaticScope::SpellsYouControl(predicate) = &grant.scope {
                    if source_controller == controller
                        && predicate.matches_card(card)
                        && grant.condition.as_ref().is_none_or(|condition| {
                            self.check_static_condition(condition, controller)
                        })
                    {
                        keywords = keywords.union(&grant.keywords);
                    }
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
                            && card.static_type_grants.is_empty()
                        {
                            return None;
                        }
                        let permanent_id = self.state.card_to_permanent[card_id]?;
                        let permanent = self.state.permanents[permanent_id].as_ref()?;
                        if self.has_lost_abilities(permanent_id) {
                            return None;
                        }
                        Some((permanent_id, permanent.controller, card))
                    })
            })
    }

    /// The characteristics an Aura attached to `permanent_id` sets, if any
    /// (the latest in battlefield order). Attachments are filtered on their
    /// own `abilities_lost` flag alone, so this never recurses.
    pub(crate) fn set_characteristics(
        &self,
        permanent_id: PermanentId,
    ) -> Option<&SetCharacteristics> {
        self.attachments_on(permanent_id)
            .filter_map(|(_, card)| card.attached_characteristics.as_ref())
            .last()
    }

    /// Whether the permanent has lost all abilities: until end of turn
    /// (`Permanent.abilities_lost`) or while an Aura sets its characteristics.
    pub(crate) fn has_lost_abilities(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.abilities_lost)
            || self.set_characteristics(permanent_id).is_some()
    }

    /// The permanent's name (CR 201.2), as set by a characteristic-setting
    /// Aura or printed.
    pub fn permanent_name(&self, permanent_id: PermanentId) -> &str {
        match self.set_characteristics(permanent_id) {
            Some(set) => &set.named,
            None => self.state.permanents[permanent_id]
                .as_ref()
                .map_or("", |permanent| &self.state.cards[permanent.card].name),
        }
    }

    /// Whether the permanent has `subtype`: printed or temporarily granted.
    /// A characteristic-setting Aura replaces its printed creature types.
    pub fn permanent_has_subtype(&self, permanent_id: PermanentId, subtype: &str) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let printed = self.state.cards[permanent.card].has_subtype(subtype);
        let printed = match self.set_characteristics(permanent_id) {
            Some(set) => {
                set.creature_types.iter().any(|s| s == subtype)
                    || (printed && NONCREATURE_SUBTYPES.contains(&subtype))
            }
            None => printed,
        };
        printed || permanent.temp_subtypes.iter().any(|s| s == subtype)
    }

    /// Card types static type grants add to `target_id` (layer 4).
    fn granted_card_types(&self, target_id: PermanentId) -> Vec<CardType> {
        let mut types = Vec::new();
        for (source_id, controller, card) in self.static_sources() {
            for grant in &card.static_type_grants {
                if self.static_scope_applies(
                    &grant.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::Types,
                ) {
                    types.extend(grant.types.iter().copied());
                }
            }
        }
        types
    }

    /// Whether a battlefield permanent is a land, printed or granted.
    pub(crate) fn permanent_is_land(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|p| self.state.cards[p.card].types.is_land())
            || self
                .granted_card_types(permanent_id)
                .contains(&CardType::Land)
    }

    /// Distinct colors among `controller`'s permanents matching `predicate`
    /// (CR 105.2; Earthen Ally's "each color among Allies you control").
    fn colors_among(&self, controller: PlayerId, predicate: &CardPredicate) -> i32 {
        self.battlefield_permanents(controller)
            .into_iter()
            .filter(|&id| self.permanent_matches_with(id, predicate, ScopeView::Keywords))
            .filter_map(|id| self.state.permanents[id].as_ref())
            .flat_map(|permanent| self.state.cards[permanent.card].colors())
            .collect::<BTreeSet<_>>()
            .len() as i32
    }

    /// Whether a static scope from `source_id` (controlled by `controller`)
    /// covers `target_id`, reading the characteristics `view` allows.
    fn static_scope_applies(
        &self,
        scope: &StaticScope,
        source_id: PermanentId,
        controller: PlayerId,
        target_id: PermanentId,
        view: ScopeView,
    ) -> bool {
        let Some(target) = self.state.permanents[target_id].as_ref() else {
            return false;
        };
        let (predicate, other) = match scope {
            StaticScope::This => return source_id == target_id,
            StaticScope::Attached => return self.attached_creature(source_id) == Some(target_id),
            StaticScope::SpellsYouControl(_) => return false,
            StaticScope::OtherYouControl(predicate) => (predicate, true),
            StaticScope::YouControl(predicate) => (predicate, false),
        };
        if target.controller != controller || (other && source_id == target_id) {
            return false;
        }
        self.permanent_matches_with(target_id, predicate, view)
    }

    /// Whether the permanent is attacking or blocking (CR 506.4).
    pub(crate) fn permanent_in_combat(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.attacking)
            || self.state.combat.as_ref().is_some_and(|combat| {
                combat
                    .attacker_to_blockers
                    .values()
                    .any(|blockers| blockers.contains(&permanent_id))
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

    /// Auras and Equipment attached to the creature `target`, in battlefield
    /// order, less those that have lost their own abilities.
    pub(crate) fn attachments_on(
        &self,
        target: PermanentId,
    ) -> impl Iterator<Item = (PermanentId, &Card)> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(move |player| self.state.zones.zone_cards(ZoneType::Battlefield, player))
            .filter_map(move |&card_id| {
                let source = self.state.card_to_permanent[card_id]?;
                (!self.state.permanents[source].as_ref()?.abilities_lost
                    && self.attached_creature(source) == Some(target))
                .then(|| (source, &self.state.cards[card_id]))
            })
    }

    /// Whether a static restriction from a battlefield permanent applies to
    /// `target_id`.
    pub(crate) fn has_restriction(&self, target_id: PermanentId, kind: Restriction) -> bool {
        self.static_sources().any(|(source_id, controller, card)| {
            card.static_restrictions.iter().any(|restriction| {
                restriction.kind == kind
                    && restriction
                        .unless
                        .as_ref()
                        .is_none_or(|condition| !self.check_static_condition(condition, controller))
                    && self.static_scope_applies(
                        &restriction.scope,
                        source_id,
                        controller,
                        target_id,
                        ScopeView::Rules,
                    )
            })
        })
    }

    /// Distinct creature types of a battlefield creature (CR 205.3m): printed
    /// subtypes of a printed creature plus temporarily granted subtypes, less
    /// noncreature subtypes. Animated lands keep only granted types.
    pub(crate) fn creature_types(&self, permanent_id: PermanentId) -> Vec<&str> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Vec::new();
        };
        let card = &self.state.cards[permanent.card];
        let printed: &[String] = if let Some(set) = self.set_characteristics(permanent_id) {
            &set.creature_types
        } else if card.types.is_creature() {
            &card.subtypes
        } else {
            &[]
        };
        let mut types: Vec<&str> = printed
            .iter()
            .chain(&permanent.temp_subtypes)
            .map(String::as_str)
            .filter(|s| !NONCREATURE_SUBTYPES.contains(s))
            .collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// The greatest number of creatures `controller` controls that share one
    /// creature type (White Lotus Tile); zero when none has a creature type.
    pub(crate) fn greatest_shared_creature_type(&self, controller: PlayerId) -> usize {
        let mut counts: BTreeMap<&str, usize> = Default::default();
        for id in self.battlefield_permanents(controller) {
            if self.permanent_is_creature(id) {
                for creature_type in self.creature_types(id) {
                    *counts.entry(creature_type).or_default() += 1;
                }
            }
        }
        counts.into_values().max().unwrap_or(0)
    }

    /// Instant timing for `player` casting `card`: printed instant or flash,
    /// or a conditional "as though it had flash" that currently holds.
    pub(crate) fn has_instant_timing(&self, player: PlayerId, card: CardId) -> bool {
        let granted_flash = self
            .exile_grant(player, card)
            .is_some_and(|grant| grant.flash);
        let card = &self.state.cards[card];
        granted_flash
            || card.is_instant_speed()
            || card
                .flash_if
                .as_ref()
                .is_some_and(|condition| self.check_static_condition(condition, player))
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
            StaticCondition::ControlsMatching { predicate } => self
                .battlefield_permanents(controller)
                .into_iter()
                .any(|id| self.permanent_matches_predicate(id, predicate)),
            StaticCondition::UnspentManaAtLeast { count } => {
                u32::from(self.pooled_mana(controller).total()) >= *count
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
        if self.has_lost_abilities(permanent_id) {
            return Keywords::default();
        }
        permanent
            .effective_keywords(&self.state.cards[permanent.card])
            .union(&self.static_keyword_bonus(permanent_id))
    }

    /// Whether the permanent can pay a {T} cost or attack (CR 302.6): it is
    /// untapped, and a creature under its controller's control since the
    /// turn began or with haste, as `effective_keywords` and
    /// `permanent_is_creature` report it. A temporary grant must not erase the
    /// restriction that applies again when the grant expires (CR 702.10).
    pub(crate) fn permanent_can_tap(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        !(permanent.tapped
            || (permanent.summoning_sick
                && self.permanent_is_creature(permanent_id)
                && !self.effective_keywords(permanent_id).haste))
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
    /// effective power, keywords and types.
    pub(crate) fn permanent_matches_predicate(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        self.permanent_matches_with(permanent_id, predicate, ScopeView::Rules)
    }

    /// Match a battlefield permanent against a predicate, counting its
    /// temporary subtypes and animation, reading the power, keywords and
    /// types `view` allows without recursing.
    fn permanent_matches_with(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
        view: ScopeView,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let card = &self.state.cards[permanent.card];
        let flying = || match view {
            ScopeView::Types | ScopeView::Keywords => {
                permanent.effective_keywords(card).flying && !self.has_lost_abilities(permanent_id)
            }
            ScopeView::PowerToughness | ScopeView::Rules => {
                self.effective_keywords(permanent_id).flying
            }
        };
        if (predicate.flying && !flying())
            || (predicate.no_counters && permanent.has_counters())
            || (predicate.in_combat && !self.permanent_in_combat(permanent_id))
        {
            return false;
        }
        if predicate.with_counter && permanent.plus1_counters <= 0 {
            return false;
        }
        if predicate.attacking
            && !self
                .state
                .combat
                .as_ref()
                .is_some_and(|combat| combat.attackers.contains(&permanent_id))
        {
            return false;
        }
        let has_subtype = |name: &str| self.permanent_has_subtype(permanent_id, name);
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
            with_counter: false,
            attacking: false,
            subtype: None,
            subtypes_any: Vec::new(),
            not_subtypes: Vec::new(),
            ..predicate.clone()
        };
        let power = match view {
            ScopeView::Rules => self.effective_power(permanent_id),
            _ => card.power.unwrap_or(0),
        };
        let mut types = Vec::new();
        if permanent.animated_creature(card) {
            types.push(CardType::Creature);
        }
        let reads_types = structural.card_type.is_some()
            || !structural.card_types_any.is_empty()
            || !structural.not_card_types.is_empty();
        if view != ScopeView::Types && reads_types {
            types.extend(self.granted_card_types(permanent_id));
        }
        structural.matches(card, power, &types)
    }
}
