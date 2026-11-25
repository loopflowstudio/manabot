// sba.rs
// State-based actions.

use crate::{
    flow::{
        decision::{EffectFrame, FrameCompletion},
        game::{Game, GameOutcome, GameProcedure, LossCauses},
        proposed_event::{EventBinding, ProposedEvent},
    },
    state::{
        ability::{Ability, TriggerCondition},
        card::{Keyword, StaticAbility},
        game_object::{ObjectId, ObjectKind, PermanentId, PlayerId},
        permanent::CounterKind,
        stack_object::StackObject,
        zone::ZoneType,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SbaProgress {
    Stable,
    Changed,
    Pending,
}

impl Game {
    /// Duplicate legendary names (layered `is_legendary` and `name`),
    /// grouped by controller in APNAP order.
    fn legend_groups(&self) -> Vec<Vec<ObjectId>> {
        let mut groups = Vec::new();
        for player in [self.active_player(), self.non_active_player()] {
            let mut names = std::collections::BTreeMap::<String, Vec<ObjectId>>::new();
            for (id, characteristics) in self.layer_views(self.iter_battlefield_permanents(player))
            {
                if characteristics
                    .static_abilities
                    .contains(&StaticAbility::LegendRuleDoesntApply)
                {
                    names.clear();
                    break;
                }
                if characteristics.is_legendary {
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    names
                        .entry(characteristics.name.clone())
                        .or_default()
                        .push(permanent.object);
                }
            }
            groups.extend(names.into_values().filter(|objects| objects.len() > 1));
        }
        groups
    }

    pub(crate) fn protected_from_loss(&self, player: PlayerId) -> bool {
        self.layer_views(self.iter_battlefield_permanents(player))
            .any(|(_, view)| {
                view.static_abilities
                    .contains(&StaticAbility::CantLoseAndOpponentsCantWin)
            })
    }

    pub fn lose_game(&mut self, player: PlayerId) {
        if !self.is_game_over() && !self.protected_from_loss(player) {
            let mut losses = [LossCauses::default(); 2];
            losses[player.0].instructed = true;
            self.state.procedure = GameProcedure::Finished(GameOutcome::Losses { losses });
        }
    }

    pub fn win_game(&mut self, winner: PlayerId) {
        if !self.is_game_over() && !self.protected_from_loss(PlayerId(1 - winner.0)) {
            self.state.procedure = GameProcedure::Finished(GameOutcome::Victory { winner });
        }
    }

    /// Evaluate both seats against the battlefield before applying this batch.
    fn commit_state_based_losses(&mut self) -> bool {
        let losses = std::array::from_fn(|seat| {
            let player = &self.state.players[seat];
            let causes = LossCauses {
                life_total: player.life <= 0,
                poison: player.counters.get(CounterKind::Poison) >= 10,
                deck_empty: player.drew_when_empty,
                ..Default::default()
            };
            if causes.any() && self.protected_from_loss(PlayerId(seat)) {
                Default::default()
            } else {
                causes
            }
        });
        // CR 704.5b considers draws only since the previous check.
        for seat in 0..2 {
            if self.state.players[seat].drew_when_empty {
                self.state.players[seat].drew_when_empty = false;
            }
        }
        if losses.iter().any(|causes| causes.any()) {
            self.state.procedure = GameProcedure::Finished(GameOutcome::Losses { losses });
            return true;
        }
        false
    }

    /// Perform one simultaneous batch, or suspend for required legend choices.
    /// Stabilization repeats changed batches before granting priority (CR 704.3).
    pub(crate) fn perform_state_based_actions(&mut self) -> SbaProgress {
        // Choose all legends before applying any of this simultaneous batch.
        // Terminal SBAs take precedence over asking a player who already lost.
        if self.is_game_over() {
            return SbaProgress::Stable;
        }
        if self.commit_state_based_losses() {
            return SbaProgress::Changed;
        }
        let groups = self.legend_groups();
        if let Some(first) = groups.first() {
            let id = self.state.permanent_of(first[0]).expect("legend");
            let player = self.state.permanents[id]
                .as_ref()
                .expect("legend")
                .controller;
            self.suspend_rule_decision(crate::flow::decision::Decision::LegendRule {
                player,
                groups,
                keep: Vec::new(),
            });
            return SbaProgress::Pending;
        }
        // No legend choice was needed, so this same simultaneous batch
        // has no legend losers. Do not discover the groups a second time.
        self.perform_state_based_batch(&[])
    }

    pub(crate) fn perform_state_based_actions_with_legends(
        &mut self,
        keep: &[ObjectId],
    ) -> SbaProgress {
        let legend_losers: Vec<ObjectId> = self
            .legend_groups()
            .into_iter()
            .flatten()
            .filter(|object| !keep.contains(object))
            .collect();
        self.perform_state_based_batch(&legend_losers)
    }

    fn perform_state_based_batch(&mut self, legend_losers: &[ObjectId]) -> SbaProgress {
        let mut performed = false;
        // CR 704.5q — +1/+1 and -1/-1 counters on one permanent annihilate
        // in pairs. Power and toughness are unchanged, so the simultaneous
        // checks below read the same characteristics either way.
        performed |= self.annihilate_opposing_counters();

        // Candidate discovery uses exact object identities. The mutable
        // storage slot is resolved only while inspecting the current object;
        // no later commit can accidentally follow a re-entered card.
        let mut candidates = Vec::new();
        for (permanent_id, characteristics) in
            self.layer_views((0..self.state.permanents.len()).map(PermanentId))
        {
            let permanent = self.state.permanents[permanent_id].as_ref().unwrap();
            // CR 704.5f — A creature with toughness 0 or less is put into
            // its owner's graveyard (an earthbent land losing its counters
            // is a 0/0 and dies).
            let zero_toughness = characteristics.is_creature() && characteristics.toughness <= 0;
            // CR 704.5i/v. No supported battle has a defeat trigger; if one
            // is introduced, its pending trigger must delay this battle SBA.
            let zero_counters = (characteristics.types.is_planeswalker()
                && permanent.counter_count(CounterKind::Loyalty) == 0)
                || (characteristics.types.is_battle()
                    && permanent.counter_count(CounterKind::Defense) == 0);
            let legend_loser = legend_losers.contains(&permanent.object);
            // CR 704.5m — An Aura attached to an illegal object or to
            // nothing is put into its owner's graveyard.
            let illegal_aura = self.aura_attachment_illegal(permanent_id);
            let put_in_graveyard = zero_toughness || zero_counters || legend_loser || illegal_aura;
            // CR 704.5g — Lethal damage destroys. The other graveyard SBAs
            // above do not destroy, even when lethal damage also applies.
            let destroy = !put_in_graveyard
                && self.has_lethal_damage(permanent_id, &characteristics)
                && !characteristics.keywords.has(Keyword::Indestructible);
            if !put_in_graveyard && !destroy {
                continue;
            }
            let Some(object_ref) = self.permanent_object(permanent_id) else {
                continue;
            };
            candidates.push((object_ref, destroy));
        }
        candidates.sort_by_key(|(object_ref, _)| *object_ref);

        let mut events: Vec<_> = candidates
            .into_iter()
            .map(|(target, destroy)| {
                if destroy {
                    ProposedEvent::Destroy { target }
                } else {
                    self.zone_move_proposal(target, ZoneType::Graveyard, Default::default())
                }
            })
            .collect();
        events.extend(
            self.finished_sagas()
                .into_iter()
                .filter_map(|id| self.permanent_object(id))
                .map(|target| ProposedEvent::Sacrifice { target }),
        );
        if !events.is_empty() {
            let batch = self.event_batch(EventBinding::Deaths, events);
            let mut frame = EffectFrame::new(self.active_player(), Vec::new());
            frame.batch = Some(batch);
            frame.completion = FrameCompletion::StateBased {
                next: Box::new(FrameCompletion::Continue),
            };
            self.run_frame(frame);
            return if self.state.suspended_decision.is_some() {
                SbaProgress::Pending
            } else {
                SbaProgress::Changed
            };
        }
        if self.finish_state_based_batch() {
            performed = true;
        }
        if performed {
            SbaProgress::Changed
        } else {
            SbaProgress::Stable
        }
    }

    fn annihilate_opposing_counters(&mut self) -> bool {
        let pairs: Vec<(PermanentId, i32)> = self
            .state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(index, permanent)| {
                let permanent = permanent.as_ref()?;
                let pairs = permanent
                    .counter_count(CounterKind::PlusOnePlusOne)
                    .min(permanent.counter_count(CounterKind::MinusOneMinusOne));
                (pairs > 0).then_some((PermanentId(index), pairs))
            })
            .collect();
        for &(id, pairs) in &pairs {
            let permanent = self.state.permanents[id].as_mut().expect("present");
            permanent.change_counters(CounterKind::PlusOnePlusOne, -pairs);
            permanent.change_counters(CounterKind::MinusOneMinusOne, -pairs);
        }
        !pairs.is_empty()
    }

    pub(crate) fn finish_state_based_batch(&mut self) -> bool {
        let mut performed = false;
        // CR 704.5n — An Equipment attached to an object that is gone or is
        // no longer a creature becomes unattached and stays on the battlefield.
        let detached: Vec<PermanentId> = (0..self.state.permanents.len())
            .map(PermanentId)
            .filter(|&id| {
                self.state.permanents[id].as_ref().is_some_and(|equipment| {
                    equipment.attached_to.is_some()
                        && self.state[equipment.object].enchant().is_none()
                }) && self
                    .attached_creature(id)
                    .is_none_or(|target| self.protected_from_permanent(target, id))
            })
            .collect();
        for permanent_id in detached {
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                permanent.attached_to = None;
            }
            performed = true;
        }

        // CR 704.5d-e — A token in a zone other than the battlefield, or a
        // spell copy outside the stack, ceases to exist. A prepared copy stays
        // in exile while its battlefield source keeps it prepared.
        // This runs after the move to the graveyard, so death triggers from
        // tokens have already been enqueued.
        let mut tokens_to_remove = Vec::new();
        for (object_id, object) in self.state.latest_objects() {
            let zone = self.state.zones.zone_of(object_id);
            let ceases = match object.kind() {
                ObjectKind::Card => false,
                ObjectKind::Token => !matches!(zone, Some(ZoneType::Battlefield) | None),
                ObjectKind::SpellCopy if zone == Some(ZoneType::Exile) => {
                    self.preparation_source(object_id).is_none()
                }
                ObjectKind::SpellCopy => !matches!(zone, Some(ZoneType::Stack) | None),
            };
            if ceases {
                tokens_to_remove.push((object_id, object.owner));
            }
        }
        for (object_id, owner) in tokens_to_remove {
            let from = self.state.zones.zone_of(object_id);
            match from {
                Some(ZoneType::Library) => {
                    self.state.library_order_pins_mut(owner).remove(&object_id);
                    // A token's departure is public. Shift known positions at
                    // that position without forgetting unrelated identities.
                    let position = self
                        .library_from_top(owner)
                        .position(|object| object == object_id)
                        .unwrap();
                    self.state
                        .known_library_mut(owner)
                        .remove_position(position);
                }
                Some(ZoneType::Hand) => {
                    self.forget_hand_card(owner, Some(self.state[object_id].definition_id));
                }
                _ => {}
            }
            self.state.zones.remove_object(object_id);
            self.assert_library_knowledge();
            self.capture_object_move(object_id, from, None, owner);
            performed = true;
        }

        performed
    }

    fn finished_sagas(&self) -> Vec<PermanentId> {
        self.layer_views((0..self.state.permanents.len()).map(PermanentId))
            .filter(|(id, view)| {
                let permanent = self.state.permanents[*id].as_ref().unwrap();
                let Some(last) = view.final_chapter() else {
                    return false;
                };
                permanent.counter_count(CounterKind::Lore) >= i32::from(last)
                    && !self.chapter_ability_pending(permanent.object)
            })
            .map(|(id, _)| id)
            .collect()
    }

    /// Whether a chapter ability of this Saga has triggered and not yet left
    /// the stack: waiting to be put on it, or on it.
    fn chapter_ability_pending(&self, saga: ObjectId) -> bool {
        let object = self.state.live(saga);
        let is_chapter = |object_id, source_ref, inline: &Option<_>, index: usize| {
            object_id == saga
                && source_ref == object
                && inline.is_none()
                && matches!(
                    self.state[saga].abilities.get(index),
                    Some(Ability::Triggered {
                        condition: TriggerCondition::Chapter { .. },
                        ..
                    })
                )
        };
        self.state.pending_triggers.iter().any(|trigger| {
            is_chapter(
                trigger.source_object(),
                trigger.source_ref(),
                &trigger.inline_effects,
                trigger.ability_index,
            )
        }) || self.state.stack_objects.iter().any(|item| {
            matches!(item, StackObject::TriggeredAbility(ability) if is_chapter(
                ability.source_object(),
                ability.source_ref(),
                &ability.inline_effects,
                ability.ability_index,
            ))
        })
    }

    /// Whether an Aura is unattached, or attached to itself or an object its
    /// enchant ability doesn't allow (CR 303.4d, 704.5m).
    fn aura_attachment_illegal(&self, aura: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[aura].as_ref() else {
            return false;
        };
        let Some(enchant) = self.state[permanent.object].enchant() else {
            return false;
        };
        !self.attached_permanent(aura).is_some_and(|object| {
            object != aura
                && !self.protected_from_permanent(object, aura)
                && self.permanent_matches_spec(object, enchant, permanent.controller)
        })
    }
}
