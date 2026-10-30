// effects.rs
// The one authority for continuous-effect records (`state::effect`): creating
// them as spells and abilities resolve, reading the rules effects they carry,
// and ending them at each duration boundary.
//
// Characteristic effects (types, keywords, ability loss, P/T) are applied with
// static abilities by layer and timestamp in `flow/statics.rs`; the rules
// queries here (blockability, death replacement, kept mana, hand size,
// firebending) read the same records. Records are journaled whole, like
// delayed triggers, so undo and rollback restore them exactly.
//
// Expiry boundaries:
//   - cleanup ends `EndOfTurn` records (CR 514.2);
//   - an affected object changing zones ends its records (CR 400.7).

use crate::{
    flow::game::Game,
    state::{
        card::CardDefId,
        effect::{Affected, ContinuousEffect, EffectDuration, Modification},
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId},
        mana::Color,
    },
};

impl Game {
    /// Record a continuous effect with a new timestamp. Every object in a set
    /// fixed at resolution (CR 611.2c) shares that timestamp.
    pub fn apply_continuous(
        &mut self,
        source: Option<CardId>,
        controller: PlayerId,
        affected: impl IntoIterator<Item = Affected>,
        modification: Modification,
        duration: EffectDuration,
    ) {
        let source: Option<CardDefId> = source.map(|card| self.state.cards[card].definition_id);
        let timestamp = self.state.id_gen.next_id();
        let records: Vec<ContinuousEffect> = affected
            .into_iter()
            .map(|affected| ContinuousEffect {
                timestamp,
                source,
                controller,
                affected,
                modification: modification.clone(),
                duration,
            })
            .collect();
        if records.is_empty() {
            return;
        }
        self.journal_effects();
        self.state.effects.extend(records);
        // Characteristics can change which mana abilities exist.
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
    }

    /// The `Affected` of a battlefield permanent's current incarnation.
    pub fn affected_permanent(&self, permanent_id: PermanentId) -> Option<Affected> {
        self.permanent_object_ref(permanent_id)
            .map(Affected::Object)
    }

    /// Records affecting the permanent's current incarnation, in timestamp
    /// order.
    pub fn effects_on(&self, permanent_id: PermanentId) -> impl Iterator<Item = &ContinuousEffect> {
        let object = self.permanent_object_ref(permanent_id);
        self.state
            .effects
            .iter()
            .filter(move |effect| object.is_some_and(|object| effect.affects_object(object)))
    }

    /// Records affecting a player, in timestamp order.
    pub fn player_effects(&self, player: PlayerId) -> impl Iterator<Item = &ContinuousEffect> {
        self.state
            .effects
            .iter()
            .filter(move |effect| effect.affects_player(player))
    }

    fn permanent_has_modification(&self, permanent_id: PermanentId, wanted: &Modification) -> bool {
        self.effects_on(permanent_id)
            .any(|effect| effect.modification == *wanted)
    }

    /// The latest timestamp at which the permanent lost all abilities, if
    /// it has (CR 613.1f); grants with later timestamps still apply.
    pub fn ability_loss_timestamp(&self, permanent_id: PermanentId) -> Option<ObjectId> {
        let recorded = self
            .effects_on(permanent_id)
            .filter(|effect| effect.modification == Modification::LoseAllAbilities)
            .map(|effect| effect.timestamp)
            .last();
        recorded.max(self.set_characteristics_timestamp(permanent_id))
    }

    /// "Can't be blocked this turn."
    pub fn cant_be_blocked(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::CantBeBlocked)
    }

    /// "If it would die this turn, exile it instead."
    pub fn exiled_if_it_would_die(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::ExileIfWouldDie)
    }

    /// Whether an effect makes the permanent an artifact creature.
    pub fn becomes_artifact_creature(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::BecomeArtifactCreature)
    }

    /// Firebending granted by effects later than any ability loss, summed.
    pub fn granted_firebending(&self, permanent_id: PermanentId) -> u8 {
        let lost = self.ability_loss_timestamp(permanent_id);
        self.effects_on(permanent_id)
            .filter(|effect| effect.follows(lost))
            .filter_map(|effect| match effect.modification {
                Modification::Firebending(amount) => Some(amount),
                _ => None,
            })
            .fold(0, u8::saturating_add)
    }

    /// Whether `player` keeps unspent mana of `color` as steps and phases end.
    pub fn keeps_unspent_mana(&self, player: PlayerId, color: Color) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::KeepUnspentMana(color))
    }

    /// Whether an effect removed `player`'s maximum hand size.
    pub fn has_no_maximum_hand_size_effect(&self, player: PlayerId) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::NoMaximumHandSize)
    }

    /// CR 514.2: "until end of turn" effects end in the cleanup step.
    pub(crate) fn end_turn_effects(&mut self) {
        self.end_effects(|effect| effect.duration == EffectDuration::EndOfTurn);
    }

    /// CR 400.7: the object that left is a new object; effects on it end.
    pub(crate) fn end_effects_on(&mut self, object: ObjectRef) {
        self.end_effects(|effect| effect.affects_object(object));
    }

    fn end_effects(&mut self, ends: impl Fn(&ContinuousEffect) -> bool) {
        if !self.state.effects.iter().any(&ends) {
            return;
        }
        self.journal_effects();
        self.state.effects.retain(|effect| !ends(effect));
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
    }
}
