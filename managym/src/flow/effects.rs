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
//   - an affected object changing zones ends its records (CR 400.7);
//   - a player's turn beginning ends the "until your next turn" records they
//     control, then their untap step phases their permanents in (CR 702.26a).

use crate::{
    cards::content_pack::default_content_pack,
    flow::game::Game,
    state::{
        card::{CardDefId, Keyword, Keywords, PrintedCharacteristics},
        effect::{Affected, ContinuousEffect, EffectDuration, Modification},
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::Color,
    },
};

impl Game {
    /// Record a continuous effect with a new timestamp. Every object in a set
    /// fixed at resolution (CR 611.2c) shares that timestamp.
    pub fn apply_continuous(
        &mut self,
        source: Option<ObjectId>,
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
        source: Option<ObjectId>,
        source_object: Option<ObjectId>,
        controller: PlayerId,
        affected: impl IntoIterator<Item = Affected>,
        modification: Modification,
        duration: EffectDuration,
    ) {
        let source_object =
            source_object.filter(|_| duration == EffectDuration::WhileYouControlSource);
        let source: Option<CardDefId> = source.map(|object| self.state[object].definition_id);
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
        source: Option<ObjectId>,
        controller: PlayerId,
    ) {
        let Some(object) = self.permanent_object(permanent_id) else {
            return;
        };
        let timestamp = self.state.id_gen_mut().next_id();
        let source: Option<CardDefId> = source.map(|object_id| self.state[object_id].definition_id);
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
        let mut records = records.into_iter().peekable();
        // Test emptiness before borrowing tracked state mutably: an empty
        // batch must preserve the derived caches.
        if records.peek().is_none() {
            return;
        }
        self.state.effects.extend(records);
        // Characteristics can change which mana abilities exist.
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
    }

    /// The `Affected` of a battlefield permanent's current incarnation.
    pub fn affected_permanent(&self, permanent_id: PermanentId) -> Option<Affected> {
        self.permanent_object(permanent_id).map(Affected::Object)
    }

    /// Records affecting the permanent's current incarnation, in timestamp
    /// order.
    pub fn effects_on(&self, permanent_id: PermanentId) -> impl Iterator<Item = &ContinuousEffect> {
        self.permanent_object(permanent_id)
            .into_iter()
            .flat_map(|object| self.state.effects.on(object))
    }

    /// The definition whose activated and triggered abilities the permanent
    /// has from its latest `HasAbilitiesOfSource` record (Koh). Callers that
    /// offer or fire those abilities check ability loss themselves.
    pub(crate) fn inherited_definition(
        &self,
        permanent_id: PermanentId,
    ) -> Option<&'static PrintedCharacteristics> {
        default_content_pack()
            .definition(self.inherited_definition_id(permanent_id)?)
            .map(|row| &*row.printed)
    }

    pub(crate) fn inherited_definition_id(&self, permanent_id: PermanentId) -> Option<CardDefId> {
        self.effects_on(permanent_id)
            .filter(|effect| effect.modification == Modification::HasAbilitiesOfSource)
            .last()?
            .source
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

    /// "Its activated abilities can't be activated this turn", mana
    /// abilities and fixed mana taps included.
    pub fn activations_prohibited(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::CantActivateAbilities)
    }

    /// "All creatures able to block [it] this turn do so."
    pub fn must_be_blocked_by_all_able(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::MustBeBlockedByAllAble)
    }

    /// "If it would die this turn, exile it instead."
    pub fn exiled_if_it_would_die(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::ExileIfWouldDie)
    }

    /// "If it would leave the battlefield, exile it instead" (unearth).
    pub fn exiled_if_it_would_leave(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::ExileIfWouldLeave)
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

    /// Whether `player`'s next creature card may be cast with flash timing.
    pub(crate) fn has_next_creature_cast_flash(&self, player: PlayerId) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::NextCreatureCastFlash)
    }

    /// The cast creature card was "the next creature card" of every such
    /// record on its caster, so all of them end (CR 601.2i).
    pub(crate) fn end_next_creature_cast_flash(&mut self, player: PlayerId) {
        self.end_effects(|effect| {
            effect.affects_player(player)
                && effect.modification == Modification::NextCreatureCastFlash
        });
    }

    /// Whether `player` skips the combat phases of this turn of theirs.
    pub(crate) fn skips_combat_phases(&self, player: PlayerId) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::SkipsCombatPhasesNextTurn)
    }

    /// The turn `player` skipped combat in is over, so its records end.
    pub(crate) fn end_skipped_combat_phases(&mut self, player: PlayerId) {
        self.end_effects(|effect| {
            effect.affects_player(player)
                && effect.modification == Modification::SkipsCombatPhasesNextTurn
        });
    }

    /// Whether `player`'s life total can't change (CR 119.7).
    pub fn life_total_cant_change(&self, player: PlayerId) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::LifeTotalCantChange)
    }

    /// Whether `player` has protection from everything (CR 702.16j).
    pub fn has_protection_from_everything(&self, player: PlayerId) -> bool {
        self.player_effects(player)
            .any(|effect| effect.modification == Modification::ProtectionFromEverything)
    }

    /// The most life `player` can pay: none other than 0 while their life
    /// total can't change (CR 119.8).
    pub(crate) fn payable_life(&self, player: PlayerId) -> i32 {
        if self.life_total_cant_change(player) {
            0
        } else {
            self.state.players[player.0].life
        }
    }

    /// Whether the permanent is phased out (CR 702.26b): the one authority
    /// battlefield readers consult. `iter_battlefield_permanents`, the layer
    /// board's sources, the trigger index, targeting and combat skip it; the
    /// observation keeps its row, marked by the record.
    #[inline]
    pub fn is_phased_out(&self, permanent_id: PermanentId) -> bool {
        self.state.effects.any_phased_out()
            && self.permanent_has_modification(permanent_id, &Modification::PhasedOut)
    }

    /// The objects a phase-out of `affected` covers: each permanent and,
    /// phasing out indirectly, everything attached to one (CR 702.26g).
    pub(crate) fn phase_out_set(&self, affected: Vec<Affected>) -> Vec<Affected> {
        let mut set = affected;
        let mut index = 0;
        while index < set.len() {
            if let Affected::Object(object) = set[index] {
                for (slot, permanent) in self.state.permanents.iter().enumerate() {
                    let Some(permanent) = permanent else { continue };
                    if permanent.attached_to != Some(object) {
                        continue;
                    }
                    let Some(attached) = self.affected_permanent(PermanentId(slot)) else {
                        continue;
                    };
                    if !set.contains(&attached) && !self.is_phased_out(PermanentId(slot)) {
                        set.push(attached);
                    }
                }
            }
            index += 1;
        }
        set
    }

    /// CR 702.26a: as `player`'s untap step begins, the permanents they
    /// control that phased out directly phase in, and with each one
    /// whatever phased out attached to it. Phasing in is no zone change: no
    /// trigger sees it, and counters and attachments stay (CR 702.26d).
    pub(crate) fn phase_in(&mut self, player: PlayerId) {
        if !self.state.effects.any_phased_out() {
            return;
        }
        let phased: Vec<(ObjectId, PlayerId, Option<ObjectId>)> = self
            .state
            .effects
            .iter()
            .filter(|effect| effect.modification == Modification::PhasedOut)
            .filter_map(|effect| match effect.affected {
                Affected::Object(object) => {
                    let id = self.lookup_current_permanent(object).ok()?;
                    let permanent = self.state.permanents[id].as_ref()?;
                    Some((object, permanent.controller, permanent.attached_to))
                }
                Affected::Player(_) => None,
            })
            .collect();
        let is_phased = |object: ObjectId| phased.iter().any(|(other, ..)| *other == object);
        let mut phasing_in: Vec<ObjectId> = phased
            .iter()
            .filter(|(_, controller, attached)| {
                *controller == player && !attached.is_some_and(is_phased)
            })
            .map(|(object, ..)| *object)
            .collect();
        let mut index = 0;
        while index < phasing_in.len() {
            let host = phasing_in[index];
            for (object, _, attached) in &phased {
                if *attached == Some(host) && !phasing_in.contains(object) {
                    phasing_in.push(*object);
                }
            }
            index += 1;
        }
        self.end_effects(|effect| {
            effect.modification == Modification::PhasedOut
                && matches!(effect.affected, Affected::Object(object) if phasing_in.contains(&object))
        });
    }

    /// `player`'s turn begins: their "until your next turn" records end.
    pub(crate) fn end_until_your_next_turn_effects(&mut self, player: PlayerId) {
        self.end_effects(|effect| {
            effect.duration == EffectDuration::UntilYourNextTurn && effect.controller == player
        });
    }

    /// CR 514.2: "until end of turn" effects end in the cleanup step.
    pub(crate) fn end_turn_effects(&mut self) {
        self.end_effects(|effect| effect.duration == EffectDuration::EndOfTurn);
    }

    /// CR 400.7: the object that left is a new object; effects on it end,
    /// as do effects lasting while their controller controls it.
    pub(crate) fn end_effects_on(&mut self, object: ObjectId) {
        self.end_effects(|effect| {
            effect.affects_object(object) || effect.source_object == Some(object)
        });
    }

    /// "For as long as you control [this]" ends when control of `object`
    /// passes to another player.
    pub(crate) fn end_effects_for_control_change(
        &mut self,
        object: ObjectId,
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
