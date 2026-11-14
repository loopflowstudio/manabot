// sba.rs
// State-based actions.

use crate::state::card::Keyword;

use crate::state::game_object::ObjectKind;
use crate::{
    flow::{
        decision::{EffectFrame, FrameCompletion},
        game::Game,
        proposed_event::{EventBinding, ProposedEvent},
    },
    state::{
        ability::{Ability, TriggerCondition},
        game_object::{ObjectId, PermanentId, PlayerId},
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
            let views: Vec<_> = self
                .layer_views(self.iter_battlefield_permanents(player))
                .collect();
            if views.iter().any(|(_, characteristics)| {
                characteristics
                    .static_abilities
                    .contains(&crate::state::card::StaticAbility::LegendRuleDoesntApply)
            }) {
                continue;
            }
            for (id, characteristics) in views {
                let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                if characteristics.is_legendary {
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

    /// Perform one simultaneous batch, or suspend for required legend choices.
    /// Stabilization repeats changed batches before granting priority (CR 704.3).
    pub(crate) fn perform_state_based_actions(&mut self) -> SbaProgress {
        // Choose all legends before applying any of this simultaneous batch.
        // Terminal SBAs take precedence over asking a player who already lost.
        if self
            .state
            .players
            .iter()
            .all(|p| p.life > 0 && !p.drew_when_empty)
        {
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
            return self.perform_state_based_batch(&[]);
        }
        self.perform_state_based_actions_with_legends(&[])
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
        for player in [PlayerId(0), PlayerId(1)] {
            // CR 704.5a, 704.5b — A player loses at 0 or less life or for drawing from empty library.
            if self.state.players[player.0].life <= 0
                || self.state.players[player.0].drew_when_empty
            {
                performed |= self.lose_game(player);
            }
        }

        if self.is_game_over() {
            return if performed {
                SbaProgress::Changed
            } else {
                SbaProgress::Stable
            };
        }

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
            let mut batch = self.event_batch(events);
            batch.binding = EventBinding::Deaths;
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
                        && self.state.objects[equipment.object].enchant().is_none()
                }) && self.attached_creature(id).is_none()
            })
            .collect();
        for permanent_id in detached {
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                permanent.attached_to = None;
            }
            performed = true;
        }

        // CR 704.5d-e — A token in a zone other than the battlefield, or a
        // copy of a spell in a zone other than the stack, ceases to exist.
        // This runs after the move to the graveyard, so death triggers from
        // tokens have already been enqueued.
        let mut tokens_to_remove = Vec::new();
        for (object_id, object) in self.state.latest_objects() {
            let zone = self.state.zones.zone_of(object_id);
            let ceases = match object.kind() {
                ObjectKind::Card => false,
                ObjectKind::Token => !matches!(zone, Some(ZoneType::Battlefield) | None),
                ObjectKind::SpellCopy => !matches!(zone, Some(ZoneType::Stack) | None),
            };
            if ceases {
                tokens_to_remove.push((object_id, object.owner));
            }
        }
        for (object_id, owner) in tokens_to_remove {
            let from = self.state.zones.zone_of(object_id);
            self.state.zones.remove_object(object_id);
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
                    self.state.objects[saga].abilities.get(index),
                    Some(Ability::Triggered {
                        condition: TriggerCondition::Chapter { .. },
                        ..
                    })
                )
        };
        self.state.pending_triggers.iter().any(|trigger| {
            is_chapter(
                trigger.source_object,
                trigger.source_ref,
                &trigger.inline_effects,
                trigger.ability_index,
            )
        }) || self.state.stack_objects.iter().any(|item| {
            matches!(item, StackObject::TriggeredAbility(ability) if is_chapter(
                ability.source_object,
                ability.source_ref,
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
        let Some(enchant) = self.state.objects[permanent.object].enchant() else {
            return false;
        };
        !self.attached_permanent(aura).is_some_and(|object| {
            object != aura && self.permanent_matches_spec(object, enchant, permanent.controller)
        })
    }

    pub(crate) fn lose_game(&mut self, player: PlayerId) -> bool {
        if !self.state.players[player.0].alive {
            return false;
        }
        self.state.players[player.0].alive = false;
        true
    }
}
