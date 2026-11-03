// effects.rs
// The one authority for continuous-effect records (`state::effect`): creating
// them as spells and abilities resolve, reading the rules effects they carry,
// and ending them at each duration boundary.
//
// Characteristic effects (types, keywords and firebending, ability loss, P/T)
// are applied with static abilities by layer and timestamp in
// `Game::characteristics` (`flow/statics.rs`); the rules queries here
// (blockability, death replacement, kept mana, hand size) read the same
// records.
//
// Expiry boundaries:
//   - cleanup ends `EndOfTurn` records (CR 514.2);
//   - an affected object changing zones ends its records (CR 400.7).

use crate::{
    flow::game::Game,
    state::{
        card::{CardDefId, Keyword, Keywords},
        effect::{Affected, ContinuousEffect, EffectDuration, Modification},
        game_object::{CardId, ObjectRef, PermanentId, PlayerId},
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
        self.record_continuous(source, None, controller, affected, modification, duration);
    }

    /// [`Game::apply_continuous`], naming the exact source object a
    /// `WhileYouControlSource` duration follows.
    pub(crate) fn record_continuous(
        &mut self,
        source: Option<CardId>,
        source_object: Option<ObjectRef>,
        controller: PlayerId,
        affected: impl IntoIterator<Item = Affected>,
        modification: Modification,
        duration: EffectDuration,
    ) {
        let source_object =
            source_object.filter(|_| duration == EffectDuration::WhileYouControlSource);
        let source: Option<CardDefId> = source.map(|card| self.state.cards[card].definition_id);
        let timestamp = self.state.id_gen_mut().next_id();
        self.push_effects(affected.into_iter().map(|affected| ContinuousEffect {
            timestamp,
            source,
            controller,
            affected,
            modification: modification.clone(),
            duration,
            source_object,
        }));
    }

    /// Earthbend's animation (CR 702.182a): the object becomes a 0/0 creature
    /// with haste in addition to its other types, as three records of one
    /// timestamp (layers 4, 7b and 6) that last as long as the object does.
    pub fn animate_permanent(
        &mut self,
        permanent_id: PermanentId,
        source: Option<CardId>,
        controller: PlayerId,
    ) {
        let Some(object) = self.permanent_object_ref(permanent_id) else {
            return;
        };
        let timestamp = self.state.id_gen_mut().next_id();
        let source: Option<CardDefId> = source.map(|card| self.state.cards[card].definition_id);
        let modifications = [
            Modification::BecomeCreature,
            Modification::SetBasePt {
                power: Some(0),
                toughness: Some(0),
            },
            Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
        ];
        self.push_effects(modifications.map(|modification| ContinuousEffect {
            timestamp,
            source,
            controller,
            affected: Affected::Object(object),
            modification,
            duration: EffectDuration::Indefinite,
            source_object: None,
        }));
    }

    fn push_effects(&mut self, records: impl IntoIterator<Item = ContinuousEffect>) {
        let records: Vec<ContinuousEffect> = records.into_iter().collect();
        if records.is_empty() {
            return;
        }
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
        self.permanent_object_ref(permanent_id)
            .into_iter()
            .flat_map(|object| self.state.effects.on(object))
    }

    /// Records affecting a player, in timestamp order.
    pub fn player_effects(&self, player: PlayerId) -> impl Iterator<Item = &ContinuousEffect> {
        self.state.effects.on_player(player)
    }

    pub(super) fn permanent_has_modification(
        &self,
        permanent_id: PermanentId,
        wanted: &Modification,
    ) -> bool {
        self.effects_on(permanent_id)
            .any(|effect| effect.modification == *wanted)
    }

    /// "Can't be blocked this turn."
    pub fn cant_be_blocked(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::CantBeBlocked)
    }

    /// "Can't block this turn."
    pub fn cant_block(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::CantBlock)
    }

    /// "It doesn't untap during its controller's untap step."
    pub fn doesnt_untap(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::DoesntUntap)
    }

    /// Keywords records remove from the permanent, whatever grants them.
    pub(crate) fn lost_keywords(&self, permanent_id: PermanentId) -> Keywords {
        self.effects_on(permanent_id)
            .filter_map(|effect| match effect.modification {
                Modification::LoseKeywords(keywords) => Some(keywords),
                _ => None,
            })
            .fold(Keywords::default(), Keywords::union)
    }

    /// "If it would die this turn, exile it instead."
    pub fn exiled_if_it_would_die(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::ExileIfWouldDie)
    }

    /// The modes the permanent's "choose one that hasn't been chosen"
    /// ability has chosen.
    pub(crate) fn chosen_modes(&self, permanent_id: PermanentId) -> Vec<usize> {
        self.effects_on(permanent_id)
            .filter_map(|effect| match effect.modification {
                Modification::ModeChosen { mode } => Some(usize::from(mode)),
                _ => None,
            })
            .collect()
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

    /// CR 400.7: the object that left is a new object; effects on it end,
    /// as do effects lasting while their controller controls it.
    pub(crate) fn end_effects_on(&mut self, object: ObjectRef) {
        self.end_effects(|effect| {
            effect.affects_object(object) || effect.source_object == Some(object)
        });
    }

    /// "For as long as you control [this]" ends when control of `object`
    /// passes to another player.
    pub(crate) fn end_effects_for_control_change(
        &mut self,
        object: ObjectRef,
        controller: PlayerId,
    ) {
        self.end_effects(|effect| {
            effect.source_object == Some(object) && effect.controller != controller
        });
    }

    fn end_effects(&mut self, ends: impl Fn(&ContinuousEffect) -> bool) {
        if !self.state.effects.iter().any(&ends) {
            return;
        }
        // A control change that ends hands its object back (CR 613.1b).
        let returned: Vec<_> = self
            .state
            .effects
            .iter()
            .filter(|effect| ends(effect))
            .filter_map(|effect| match (effect.affected, &effect.modification) {
                (Affected::Object(object), Modification::ControlChanged { previous }) => {
                    Some((object, effect.controller, *previous))
                }
                _ => None,
            })
            .collect();
        self.state.effects.retain(|effect| !ends(effect));
        for (object, holder, previous) in returned {
            if let Ok(id) = self.lookup_current_permanent(object) {
                if self.state.permanents[id]
                    .as_ref()
                    .is_some_and(|permanent| permanent.controller == holder)
                {
                    self.change_control(id, previous);
                }
            }
        }
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
    }
}
