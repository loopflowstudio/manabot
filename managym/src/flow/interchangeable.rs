//! Equivalence classes of game-indistinguishable objects and action-space
//! folding.
//!
//! Two objects are interchangeable when they have identical object/permanent
//! state (modulo unique ids, settled `spell_entry` payment receipts, and plain
//! non-animating land summoning sickness / turn of entry), neither is face
//! down, neither has a layer-ordering timestamp (static ability or keyword
//! counter), and neither is referenced anywhere in live rules state outside its
//! own zone slot, its own permanent record, or the current decision's candidate
//! list.
//!
//! Fail-closed by construction: the reference census and permanent comparator
//! destructure every live rules struct without `..`, so adding a new
//! id-bearing field fails to compile until it is explicitly classified.

use crate::{
    agent::action::{Action, ActionSpace},
    flow::{
        cast_permission::{CastPermissions, PlayGrant, PlayRoute},
        combat::CombatState,
        decision::{
            Decision, EffectFrame, FrameCompletion, ManaProduction, PendingManaBonus,
            ResolutionCastOffer, ResolutionTarget, SpendCompletion, SuspendedResolution,
            TargetIteration,
        },
        event::{CombatPlayerHit, DamageTarget, GameEvent},
        game::{ActivationSource, CastChoices, Game, GameProcedure, GameState, PendingChoice},
        payment::{CostComponent, Help, Operation, Payment, PaymentProgress},
        priority::PriorityState,
        proposed_event::{
            DiscardCause, EntryFace, EventBatch, EventBinding, PendingEvent, ProposedDamageTarget,
            ProposedEvent, ReplacementId, ZoneEntry,
        },
        statics::Characteristics,
        trigger::{
            DelayedTrigger, DelayedTriggerEvent, ExileLink, PendingTrigger, TriggerPlacement,
        },
        turn::{RecordedCast, TurnState},
    },
    state::{
        ability::Effect,
        binding::ObjectBinding,
        card::{AbilityGrant, CardDefId, CardType, SpellProfile, StaticAbility},
        effect::{Affected, ContinuousEffect, Modification},
        game_object::{CardId, ObjectId, ObjectKind, ObjectLki, PermanentId, PlayerId, Target},
        permanent::{Permanent, SpellEntry},
        player::Player,
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        zone::ZoneType,
    },
};

/// Folded members of surviving actions: `(surviving_action_index, folded_object)`
/// pairs, stored beside `Game::current_action_space` and absent when nothing
/// folded.
pub(crate) type ActionAliases = Option<Box<[(usize, ObjectId)]>>;

/// Flat, stack-first reference census over a small watched set of candidate
/// objects. Walks all live rules state once and marks any watched object named
/// anywhere in it.
struct ReferenceCensus<'a> {
    state: &'a GameState,
    watch: &'a [ObjectId],
    named_mask: u64,
    overflow: Vec<bool>,
}

impl<'a> ReferenceCensus<'a> {
    fn run(game: &'a Game, watch: &'a [ObjectId]) -> Self {
        let mut census = Self {
            state: &game.state,
            watch,
            named_mask: 0,
            overflow: if watch.len() > 64 {
                vec![false; watch.len() - 64]
            } else {
                Vec::new()
            },
        };
        if !watch.is_empty() {
            census.walk_game(game);
        }
        census
    }

    #[inline]
    fn is_named(&self, index: usize) -> bool {
        if index < 64 {
            (self.named_mask & (1u64 << index)) != 0
        } else {
            self.overflow[index - 64]
        }
    }

    #[inline]
    fn all_named(&self) -> bool {
        let len = self.watch.len();
        if len <= 64 {
            let full = if len == 64 {
                u64::MAX
            } else {
                (1u64 << len) - 1
            };
            self.named_mask == full
        } else {
            self.named_mask == u64::MAX && self.overflow.iter().all(|&b| b)
        }
    }

    #[inline]
    fn mark_object(&mut self, object: ObjectId) {
        for (i, &watched) in self.watch.iter().enumerate() {
            if watched == object {
                if i < 64 {
                    self.named_mask |= 1u64 << i;
                } else {
                    self.overflow[i - 64] = true;
                }
            }
        }
    }

    #[inline]
    fn mark_opt_object(&mut self, object: Option<ObjectId>) {
        if let Some(object) = object {
            self.mark_object(object);
        }
    }

    #[inline]
    fn mark_permanent(&mut self, id: PermanentId) {
        if let Some(permanent) = self.state.permanents.get(id.0).and_then(Option::as_ref) {
            self.mark_object(permanent.object);
        }
    }

    #[inline]
    fn mark_opt_permanent(&mut self, id: Option<PermanentId>) {
        if let Some(id) = id {
            self.mark_permanent(id);
        }
    }

    #[inline]
    fn mark_card(&mut self, card: CardId) {
        self.mark_object(self.state.object_of(card));
    }

    #[inline]
    fn mark_target(&mut self, target: Target) {
        match target {
            Target::Player(_) | Target::StackAbility(_) => {}
            Target::Permanent(id) => self.mark_permanent(id),
            Target::StackSpell(object) | Target::GraveyardCard(object) => {
                self.mark_object(object);
            }
        }
    }

    #[inline]
    fn mark_opt_target(&mut self, target: Option<Target>) {
        if let Some(target) = target {
            self.mark_target(target);
        }
    }

    fn walk_game(&mut self, game: &Game) {
        if let Some(choice) = &game.pending_choice {
            self.walk_pending_choice(choice);
        }
        self.walk_game_state(&game.state);
    }

    fn walk_game_state(&mut self, state: &GameState) {
        let GameState {
            procedure,
            disclosure: _, // decklist visibility enum, no ids
            cards: _,      // immutable physical card roster
            objects: _,    // object storage
            permanents,
            object_to_permanent: _, // storage index
            object_lki: _,          // vouched history (departed objects)
            spell_lki: _,           // vouched history (departed spells)
            players,
            zones: _,              // zone storage
            library_order_pins: _, // vouched: private branch sampling only, rules never read it
            turn,
            priority,
            stack_objects,
            combat,
            events: _, // vouched history
            pending_events,
            observation_events: _, // vouched history/projection
            pending_triggers,
            trigger_placement,
            delayed_triggers,
            exile_links,
            object_counters,
            effects,
            cast_permissions,
            suspended_decision,
            cast_caller,
            trigger_enqueue_counter: _,
            rng: _,
            id_gen: _,
        } = state;

        // Procedure holds no object ids (OpeningHands is counts/stage, Finished is outcome).
        match procedure {
            GameProcedure::OpeningHands(_)
            | GameProcedure::Playing
            | GameProcedure::Finished(_) => {}
        }

        // PriorityState holds no object ids.
        let PriorityState {
            holder: _,
            consecutive_passes: _,
            pass_count: _,
            sba_done: _,
            loop_watch: _,
            resolution_owed: _,
        } = priority;

        // Battlefield permanents: only `attached_to` names another object.
        // A settled permanent's `spell_entry.payment` is vouched as history.
        for permanent in permanents.iter().flatten() {
            let Permanent {
                transforms: _,
                manifested: _,
                id: _,
                object: _, // own slot record
                controller: _,
                control_differs_from_entry: _,
                tapped: _,
                summoning_sick: _,
                damage: _,
                deathtouch_damage: _,
                dealt_damage_this_turn: _,
                entry_turn: _,
                counters: _,
                keyword_counter_timestamps: _,
                timestamp: _,
                attacking: _,
                spell_entry: _, // vouched as history for settled permanents
                attached_to,
                spent_abilities: _,
                activated_this_turn: _,
            } = permanent;
            self.mark_opt_object(*attached_to);
        }

        for player in players {
            let Player {
                id: _,
                index: _,
                deck: _,          // immutable roster
                sideboard: _,     // immutable roster
                known_hand: _,    // definition counts only
                known_library: _, // definition counts by rank only
                name: _,
                life: _,
                starting_life: _,
                counters: _,
                drew_when_empty: _,
                mana_pool,
                combat_mana_pool,
            } = player;
            mana_pool.visit_referenced_objects(|obj| self.mark_object(obj));
            combat_mana_pool.visit_referenced_objects(|obj| self.mark_object(obj));
        }

        self.walk_turn_state(turn);

        for stack_object in stack_objects {
            self.walk_stack_object(stack_object);
        }

        if let Some(combat) = combat.as_deref() {
            self.walk_combat_state(combat);
        }

        for event in pending_events {
            self.walk_game_event(event);
        }

        for trigger in pending_triggers {
            self.walk_pending_trigger(trigger);
        }

        if let Some(placement) = trigger_placement.as_deref() {
            let TriggerPlacement {
                remaining,
                active: _,
                group: _,
                placing,
            } = placement;
            for trigger in remaining {
                self.walk_pending_trigger(trigger);
            }
            if let Some(trigger) = placing {
                self.walk_pending_trigger(trigger);
            }
        }

        for delayed in delayed_triggers {
            self.walk_delayed_trigger(delayed);
        }

        for link in exile_links {
            let ExileLink {
                source,
                exiled_object,
                returns: _,
            } = link;
            self.mark_object(*source);
            self.mark_object(*exiled_object);
        }

        for &object in object_counters.keys() {
            self.mark_object(object);
        }

        for effect in effects.iter() {
            let ContinuousEffect {
                timestamp: _,
                source: _,
                controller: _,
                affected,
                modification: _,
                duration: _,
                source_object,
            } = effect;
            match affected {
                Affected::Player(_) => {}
                Affected::Object(object) => self.mark_object(*object),
            }
            self.mark_opt_object(*source_object);
        }

        self.walk_cast_permissions(cast_permissions);

        if let Some(suspended) = suspended_decision {
            self.walk_suspended(suspended, true);
        }

        if let Some(caller) = cast_caller.as_deref() {
            self.walk_suspended(caller, false);
        }
    }

    fn walk_turn_state(&mut self, turn: &TurnState) {
        let TurnState {
            player_controls: _,
            active_player: _,
            turn_number: _,
            lands_played: _,
            spells_cast,
            history: _,
            graveyard_from_battlefield,
            damage_pairs,
            attacks_this_turn: _, // per-object count compared in `same_permanent_state_cheap`
            ability_triggers_this_turn,
            used_trigger_grants,
            ability_resolutions_this_turn,
            extra_turns: _,
            resume_with: _,
            additional_combats: _,
            combat_attackers: _,
            current_phase: _,
            current_step: _,
            step_initialized: _,
            turn_based_actions_complete: _,
            cleanup_priority: _,
        } = turn;

        for seat_casts in spells_cast {
            for RecordedCast { object, profile: _ } in seat_casts {
                self.mark_object(*object);
            }
        }
        for &object in graveyard_from_battlefield {
            self.mark_object(object);
        }
        damage_pairs.visit_objects(|obj| self.mark_object(obj));
        for &(object, _, _) in ability_triggers_this_turn.keys() {
            self.mark_object(object);
        }
        used_trigger_grants.visit_objects(|obj| self.mark_object(obj));
        for &(object, _, _) in ability_resolutions_this_turn.keys() {
            self.mark_object(object);
        }
    }

    fn walk_combat_state(&mut self, combat: &CombatState) {
        let CombatState {
            first_strikers,
            regular_damage_step: _,
            attackers,
            attacker_to_blockers,
            already_attacking,
            blocked,
            attackers_to_declare: _, // vouched candidate list for DeclareAttackers
            blockers_to_declare: _,  // vouched candidate list for DeclareBlockers
            declined_attackers: _,   // vouched symmetry-breaking list for DeclareAttackers
            declined_blockers: _,    // vouched symmetry-breaking list for DeclareBlockers
        } = combat;
        for &id in first_strikers {
            self.mark_permanent(id);
        }
        // Attackers already in combat before declaration (CR 508.4) do not fire
        // `AttackersDeclared` triggers at the end of declaration; mark them so
        // they never fold with newly declared attackers. Unblocked attackers
        // with no blockers yet compare by combat role in `same_combat_role`.
        for &id in attackers.iter().take(*already_attacking) {
            self.mark_permanent(id);
        }
        for &id in blocked {
            self.mark_permanent(id);
        }
        for (&attacker, blockers) in attacker_to_blockers {
            if !blockers.is_empty() {
                self.mark_permanent(attacker);
                for &blocker in blockers {
                    self.mark_permanent(blocker);
                }
            }
        }
    }

    fn walk_cast_permissions(&mut self, perms: &CastPermissions) {
        let CastPermissions {
            grants,
            hand_only: _,
            spent_types,
        } = perms;
        for grant in grants {
            let PlayGrant {
                key: _,
                origin: _,
                object,
                player: _,
                play_lands: _,
                cost: _,
                this_turn: _,
                through_next_turn: _,
                source,
                own_turn_only: _,
                waterbend: _,
                flash: _,
                any_mana: _,
                after_turn: _,
                foretold: _,
                on_adventure: _,
                follows_source_controller: _,
                flashback: _,
                prepared: _,
                face_down: _,
                condition: _,
            } = grant;
            self.mark_object(*object);
            self.mark_opt_object(*source);
        }
        for &(_, route) in spent_types {
            self.walk_play_route(route);
        }
    }

    fn walk_play_route(&mut self, route: PlayRoute) {
        match route {
            PlayRoute::Hand
            | PlayRoute::FreeFromHand
            | PlayRoute::LibraryTop
            | PlayRoute::PrintedFlashback
            | PlayRoute::GrantedFlashback
            | PlayRoute::Grant(_)
            | PlayRoute::ExileCounter(_)
            | PlayRoute::PrintedGraveyard => {}
            PlayRoute::GraveyardType {
                source,
                permanent_id: _,
                kind: _,
            } => self.mark_object(source),
        }
    }

    fn walk_cast_choices(&mut self, cast: &CastChoices) {
        let CastChoices {
            exile_instead_of_graveyard: _,
            profile: _,
            route,
            flashback: _,
            from_prepared: _,
            flash_surcharge: _,
            kicked: _,
            x: _,
            free: _,
            resolving: _,
        } = cast;
        if let Some(route) = route {
            self.walk_play_route(*route);
        }
    }

    fn walk_activation_source(&mut self, source: ActivationSource) {
        match source {
            ActivationSource::Permanent(id) => self.mark_permanent(id),
            ActivationSource::Card(card) => self.mark_card(card),
        }
    }

    fn walk_object_binding(&mut self, binding: &ObjectBinding) {
        let ObjectBinding {
            object,
            selection: _,
            known_to: _,
        } = binding;
        self.mark_object(*object);
    }

    fn walk_object_lki(&mut self, lki: &ObjectLki) {
        let ObjectLki {
            copiable: _,
            transforms: _,
            inherited_from: _,
            granted_abilities: _,
            keywords: _,
            creature_types: _,
            permanent,
            battlefield: _,
            attached_to,
            meld_partner,
            abilities_lost: _,
            dealt_damage_this_turn: _,
            power: _,
            toughness: _,
            counters: _,
            object,
            from_zone: _,
            owner: _,
            controller: _,
            definition_id: _,
            presentation_id: _,
        } = lki;
        self.mark_permanent(*permanent);
        self.mark_opt_object(*attached_to);
        self.mark_opt_object(*meld_partner);
        self.mark_object(*object);
    }

    fn walk_stack_object(&mut self, stack_object: &StackObject) {
        match stack_object {
            StackObject::Spell(spell) => {
                let SpellOnStack {
                    exile_instead_of_graveyard: _,
                    cast_from: _,
                    flashback: _,
                    from_prepared: _,
                    profile: _,
                    id: _,
                    object,
                    controller: _,
                    source_definition_id: _,
                    targets,
                    target_req_indices: _,
                    kicked: _,
                    x: _,
                    mode: _,
                    spent,
                } = spell;
                self.mark_object(*object);
                for &target in targets {
                    self.mark_target(target);
                }
                spent.visit_referenced_objects(|obj| self.mark_object(obj));
            }
            StackObject::ActivatedAbility(ability) => {
                self.walk_activated_ability_on_stack(ability);
            }
            StackObject::TriggeredAbility(trigger) => {
                let TriggeredAbilityOnStack {
                    source_face: _,
                    program: _,
                    triggering_amount: _,
                    remembered,
                    source_transforms: _,
                    id: _,
                    controller: _,
                    source_identity,
                    source_lki,
                    source_definition_id: _,
                    ability_index: _,
                    targets,
                    target_req_indices: _,
                    context,
                    player: _,
                    kicked: _,
                    x: _,
                    spent,
                    chosen_modes: _,
                    inline_effects: _,
                } = trigger;
                for binding in remembered {
                    self.walk_object_binding(binding);
                }
                self.mark_object(source_identity.object());
                if let Some(lki) = source_lki {
                    self.walk_object_lki(lki);
                }
                for &target in targets {
                    self.mark_target(target);
                }
                self.mark_opt_target(*context);
                spent.visit_referenced_objects(|obj| self.mark_object(obj));
            }
        }
    }

    fn walk_activated_ability_on_stack(&mut self, ability: &ActivatedAbilityOnStack) {
        let ActivatedAbilityOnStack {
            effects: _,
            program: _,
            source_transforms: _,
            inherited_from: _,
            id: _,
            controller: _,
            source_definition_id: _,
            source_object,
            ability_index: _,
            targets,
            target_req_indices: _,
            attached,
            x: _,
            granted_by,
            paid,
        } = ability;
        self.mark_object(*source_object);
        for &target in targets {
            self.mark_target(target);
        }
        self.mark_opt_object(*attached);
        if let Some(AbilityGrant { grantor, index: _ }) = granted_by {
            self.mark_object(*grantor);
        }
        paid.visit_referenced_objects(|obj| self.mark_object(obj));
    }

    fn walk_pending_trigger(&mut self, trigger: &PendingTrigger) {
        let PendingTrigger {
            program: _,
            triggering_amount: _,
            remembered,
            source_identity,
            targets,
            target_req_indices: _,
            requirement_index: _,
            source_lki,
            ability_index: _,
            controller: _,
            id: _,
            ordering_class: _,
            context,
            player: _,
            kicked: _,
            x: _,
            spent,
            attacker,
            chosen_modes: _,
            modes_chosen: _,
            inline_effects: _,
        } = trigger;
        for binding in remembered {
            self.walk_object_binding(binding);
        }
        self.mark_object(source_identity.object());
        for &target in targets {
            self.mark_target(target);
        }
        if let Some(lki) = source_lki {
            self.walk_object_lki(lki);
        }
        self.mark_opt_target(*context);
        spent.visit_referenced_objects(|obj| self.mark_object(obj));
        self.mark_opt_permanent(*attacker);
    }

    fn walk_delayed_trigger(&mut self, delayed: &DelayedTrigger) {
        let DelayedTrigger {
            remembered,
            event,
            controller: _,
            source_identity,
            effects: _,
            frequency: _,
            expiry: _,
        } = delayed;
        for binding in remembered {
            self.walk_object_binding(binding);
        }
        match event {
            DelayedTriggerEvent::Leaves {
                watched,
                or_exiled: _,
            } => self.mark_object(*watched),
            DelayedTriggerEvent::CreatureDies
            | DelayedTriggerEvent::EndStep { .. }
            | DelayedTriggerEvent::YouCastSpell(_)
            | DelayedTriggerEvent::Upkeep => {}
        }
        self.mark_object(source_identity.object());
    }

    fn walk_game_event(&mut self, event: &GameEvent) {
        match event {
            GameEvent::PermanentSacrificed {
                object_id,
                controller: _,
            } => self.mark_object(*object_id),
            GameEvent::CardRevealed { .. }
            | GameEvent::LifeChanged { .. }
            | GameEvent::CardDrawn { .. }
            | GameEvent::Bent { .. }
            | GameEvent::Scried { .. }
            | GameEvent::Surveilled { .. }
            | GameEvent::LibrarySearched { .. }
            | GameEvent::CardsLeftGraveyard { .. }
            | GameEvent::TurnStarted { .. }
            | GameEvent::StepStarted { .. }
            | GameEvent::CombatAttackersDeclared { .. }
            | GameEvent::BlockersDeclared { .. }
            | GameEvent::CombatDamageDealt { .. }
            | GameEvent::PermanentsDied { .. } => {}
            GameEvent::CardMoved {
                object_id,
                departed,
                from: _,
                to: _,
                controller: _,
                discarded: _,
                face_down: _,
            } => {
                self.mark_object(*object_id);
                self.mark_opt_object(*departed);
            }
            GameEvent::DamageDealt {
                source,
                target,
                amount: _,
                combat: _,
            } => {
                self.mark_opt_object(*source);
                match target {
                    DamageTarget::Player(_) => {}
                    DamageTarget::Permanent(id) => self.mark_permanent(*id),
                }
            }
            GameEvent::SpellCast {
                object_id,
                targets,
                caster: _,
                spend_before: _,
                spend_after: _,
            } => {
                self.mark_object(*object_id);
                for &target in targets {
                    self.mark_target(target);
                }
            }
            GameEvent::SpellResolved { object_id } => self.mark_object(*object_id),
            GameEvent::SpellCountered { object_id, by } => {
                self.mark_object(*object_id);
                self.mark_opt_object(*by);
            }
            GameEvent::AbilityCountered {
                ability: _,
                source_object,
                controller: _,
                by,
            } => {
                self.mark_object(*source_object);
                self.mark_opt_object(*by);
            }
            GameEvent::AbilityTriggered {
                source_object,
                controller: _,
                ability: _,
                attacker,
            } => {
                self.mark_object(*source_object);
                self.mark_opt_permanent(*attacker);
            }
            GameEvent::PermanentTapped {
                permanent,
                for_mana: _,
            }
            | GameEvent::PermanentUntapped { permanent }
            | GameEvent::CountersPut { permanent, .. } => self.mark_permanent(*permanent),
            GameEvent::PermanentTargeted {
                permanent,
                by,
                by_controller: _,
            } => {
                self.mark_permanent(*permanent);
                self.mark_target(*by);
            }
            GameEvent::AttackersDeclared {
                player: _,
                attackers,
            } => {
                for &id in attackers {
                    self.mark_permanent(id);
                }
            }
            GameEvent::CombatDamageToPlayers { hits } => {
                for CombatPlayerHit {
                    source,
                    recipient: _,
                    amount: _,
                } in hits
                {
                    self.mark_object(*source);
                }
            }
            GameEvent::SourceDealtDamage {
                source,
                amount: _,
                combat: _,
            } => self.mark_object(*source),
            GameEvent::CombatDamageTotals { dealt } => {
                for &(object, _) in dealt {
                    self.mark_object(object);
                }
            }
        }
    }

    fn walk_pending_choice(&mut self, choice: &PendingChoice) {
        match choice {
            PendingChoice::Pay { player: _, payment } => self.walk_payment(payment),
            PendingChoice::ActivatedTarget {
                player: _,
                source,
                ability_index: _,
                chosen,
                requirement: _,
            } => {
                self.walk_activation_source(*source);
                for &target in chosen {
                    self.mark_target(target);
                }
            }
            PendingChoice::AnnounceX {
                player: _,
                object,
                cast,
            }
            | PendingChoice::KickerChoice {
                player: _,
                object,
                cast,
            }
            | PendingChoice::SpellMode {
                player: _,
                object,
                cast,
            } => {
                self.mark_object(*object);
                self.walk_cast_choices(cast);
            }
            PendingChoice::ActivationX {
                player: _,
                permanent,
                ability_index: _,
            } => self.mark_permanent(*permanent),
            PendingChoice::ChooseTargets {
                player: _,
                object,
                cast,
                mode: _,
                requirement_index: _,
                chosen,
                chosen_req_indices: _,
                legal_targets: _, // vouched candidate list for the current decision
            } => {
                self.mark_object(*object);
                self.walk_cast_choices(cast);
                for &target in chosen {
                    self.mark_target(target);
                }
            }
        }
    }

    fn walk_payment(&mut self, payment: &Payment) {
        let Payment {
            operation,
            owed,
            receipt,
        } = payment;
        match operation {
            Operation::Spell {
                object,
                cast,
                mode: _,
                targets,
                target_req_indices: _,
            } => {
                self.mark_object(*object);
                self.walk_cast_choices(cast);
                for &target in targets {
                    self.mark_target(target);
                }
            }
            Operation::Activation {
                source,
                ability_index: _,
                targets,
                x: _,
            } => {
                self.walk_activation_source(*source);
                for &target in targets {
                    self.mark_target(target);
                }
            }
        }
        for component in owed {
            self.walk_cost_component(component);
        }
        receipt.visit_referenced_objects(|obj| self.mark_object(obj));
    }

    fn walk_cost_component(&mut self, component: &CostComponent) {
        match component {
            CostComponent::TapPower { .. }
            | CostComponent::Tap(..)
            | CostComponent::Discard
            | CostComponent::ExileFromHand
            | CostComponent::Sacrifice(..)
            | CostComponent::Forage { .. }
            | CostComponent::Life(_)
            | CostComponent::RemoveSourceCounters(..)
            | CostComponent::AddCounters(..)
            | CostComponent::Source(_) => {}
            CostComponent::Mana { cost: _, help } => {
                let Help {
                    waterbend: _,
                    convoke: _,
                    improvise: _,
                    delve,
                } = help;
                self.mark_opt_object(*delve);
            }
            CostComponent::TapPays { tapped, options: _ } => self.mark_object(*tapped),
            CostComponent::RemoveCounters {
                remaining: _,
                choosing_from,
            } => self.mark_opt_object(*choosing_from),
            CostComponent::ExileGrantor(object) | CostComponent::TapGrantor(object) => {
                self.mark_object(*object);
            }
        }
    }

    fn walk_payment_progress(&mut self, progress: &PaymentProgress) {
        let PaymentProgress {
            committed: _,
            player: _,
            owed,
            receipt,
            if_paid: _,
            if_declined: _,
            spends,
            counters_on_decline,
        } = progress;
        self.walk_cost_component(owed);
        receipt.visit_referenced_objects(|obj| self.mark_object(obj));
        if let Some((object, _, _)) = spends {
            self.mark_object(*object);
        }
        self.mark_opt_target(*counters_on_decline);
    }

    fn walk_suspended(&mut self, suspended: &SuspendedResolution, is_current: bool) {
        let SuspendedResolution { frame, decision } = suspended;
        self.walk_effect_frame(frame);
        self.walk_decision(decision, is_current);
    }

    fn walk_effect_frame(&mut self, frame: &EffectFrame) {
        let EffectFrame {
            program: _,
            batch,
            iteration,
            remembered,
            inherited_from: _,
            source_transforms: _,
            source_identity,
            controller: _,
            resolving_controller: _,
            resolutions_this_turn: _,
            triggering_amount: _,
            kicked: _,
            x: _,
            spent,
            excess_damage: _,
            life_lost: _,
            this_way,
            selected,
            selected_from: _,
            moved,
            selected_revealed: _,
            chosen,
            created,
            targets,
            active_requirement: _,
            context_target,
            attached,
            triggering_player: _,
            grantor,
            ability: _,
            ability_face: _,
            queue: _,
            completion,
        } = frame;

        if let Some(batch) = batch {
            self.walk_event_batch(batch);
        }
        if let Some(TargetIteration {
            remaining,
            effects: _,
        }) = iteration
        {
            for ResolutionTarget {
                target,
                requirement: _,
                legal: _,
            } in remaining
            {
                self.mark_target(*target);
            }
        }
        for binding in remembered {
            self.walk_object_binding(binding);
        }
        if let Some(source) = source_identity {
            self.mark_object(source.object());
        }
        spent.visit_referenced_objects(|obj| self.mark_object(obj));
        for &object in this_way {
            self.mark_object(object);
        }
        for &object in selected {
            self.mark_object(object);
        }
        for &object in moved {
            self.mark_object(object);
        }
        self.mark_opt_object(*chosen);
        for &object in created {
            self.mark_object(object);
        }
        for ResolutionTarget {
            target,
            requirement: _,
            legal: _,
        } in targets
        {
            self.mark_target(*target);
        }
        self.mark_opt_target(*context_target);
        self.mark_opt_object(*attached);
        self.mark_opt_object(*grantor);
        self.walk_frame_completion(completion);
    }

    fn walk_frame_completion(&mut self, completion: &FrameCompletion) {
        match completion {
            FrameCompletion::Priority | FrameCompletion::Continue | FrameCompletion::EndTurn => {}
            FrameCompletion::TurnFaceUp(object) => self.mark_object(*object),
            FrameCompletion::StateBased { next } => self.walk_frame_completion(next),
            FrameCompletion::DiscardPayment(payment) => self.walk_payment(payment),
            FrameCompletion::DiscardDuringResolution(caller) => self.walk_suspended(caller, false),
            FrameCompletion::Paid { player: _, payment } => self.walk_payment(payment),
            FrameCompletion::Activated {
                ability,
                source_after_cost,
            } => {
                self.walk_activated_ability_on_stack(ability);
                self.mark_opt_object(*source_after_cost);
            }
            FrameCompletion::Frame(frame) => self.walk_effect_frame(frame),
            FrameCompletion::Event { event, frame } => {
                self.walk_game_event(event);
                self.walk_effect_frame(frame);
            }
            FrameCompletion::Decision(caller) => self.walk_suspended(caller, false),
            FrameCompletion::Spell { object, attach_to } => {
                self.mark_object(*object);
                self.mark_opt_object(*attach_to);
            }
        }
    }

    fn walk_event_batch(&mut self, batch: &EventBatch) {
        let EventBatch {
            lifelink_proposed: _,
            events,
            cursor: _,
            binding,
        } = batch;
        for pending in events {
            let PendingEvent {
                event,
                ordinal: _,
                sacrifice,
                applied,
                copy_chosen: _,
                second_card_first: _,
            } = pending;
            match event {
                ProposedEvent::Damage {
                    source,
                    target,
                    amount: _,
                    combat: _,
                } => {
                    self.mark_opt_object(*source);
                    match target {
                        ProposedDamageTarget::Player(_) => {}
                        ProposedDamageTarget::Object(object) => self.mark_object(*object),
                    }
                }
                ProposedEvent::LifeChange { .. } => {}
                ProposedEvent::Destroy { target }
                | ProposedEvent::Sacrifice { target }
                | ProposedEvent::CounterChange { target, .. } => self.mark_object(*target),
                ProposedEvent::CreateTokens {
                    definition: _,
                    controller: _,
                    count: _,
                    entry,
                } => self.walk_zone_entry(entry),
                ProposedEvent::ZoneMove {
                    object_id,
                    from: _,
                    to: _,
                    entry,
                } => {
                    self.mark_object(*object_id);
                    self.walk_zone_entry(entry);
                }
            }
            if let Some((object, _)) = sacrifice {
                self.mark_object(*object);
            }
            for ReplacementId {
                source,
                definition_index: _,
                static_ability: _,
                effect: _,
            } in applied
            {
                self.mark_object(*source);
            }
        }
        match binding {
            EventBinding::None
            | EventBinding::ThisWay
            | EventBinding::Discard { .. }
            | EventBinding::Destroyed
            | EventBinding::LifeLost
            | EventBinding::Damage { .. }
            | EventBinding::Deaths
            | EventBinding::Combat
            | EventBinding::PlaceInLibrary { .. }
            | EventBinding::Shuffle { .. }
            | EventBinding::Bending { .. }
            | EventBinding::Created { .. } => {}
            EventBinding::Countered { by } => self.mark_opt_object(*by),
            EventBinding::Moved {
                linked,
                shuffle: _,
                library_end: _,
                reveal: _,
            } => {
                if let Some((object, _)) = linked {
                    self.mark_object(*object);
                }
            }
            EventBinding::ExileLinked {
                actor: _,
                grantee: _,
                source,
                grant: _,
            } => self.mark_opt_object(*source),
        }
    }

    fn walk_zone_entry(&mut self, entry: &ZoneEntry) {
        let ZoneEntry {
            discard,
            chosen_creature_type: _,
            cast_from: _,
            resolving_adventure: _,
            adventure_player: _,
            profile: _,
            prepared: _,
            shuffle_library: _,
            tapped: _,
            counters: _,
            spell,
            controller: _,
            face_down: _,
            face,
            entry_copy: _,
            attached_to,
            library_top_departure: _,
            inspected_library_departure: _,
        } = entry;
        if let Some(cause) = discard {
            match cause {
                DiscardCause::Effect {
                    controller: _,
                    source,
                } => self.mark_opt_object(*source),
                DiscardCause::Cost(_) | DiscardCause::Cleanup => {}
            }
        }
        if let Some(spell) = spell.as_deref() {
            let SpellEntry {
                kicked: _,
                x: _,
                payment,
            } = spell;
            if let Some(receipt) = payment {
                receipt.visit_referenced_objects(|obj| self.mark_object(obj));
            }
        }
        match face {
            EntryFace::Front
            | EntryFace::Transformed
            | EntryFace::ModalBack
            | EntryFace::Manifested => {}
            EntryFace::Melded { partner, result: _ } => self.mark_card(*partner),
        }
        self.mark_opt_object(*attached_to);
    }

    fn walk_decision(&mut self, decision: &Decision, is_current: bool) {
        match decision {
            Decision::OrderMeldedCards { player: _, cards } => {
                for &card in cards {
                    self.mark_card(card);
                }
            }
            Decision::AssignCombatDamage {
                player: _,
                attacker,
                blocker,
                minimum: _,
                remaining: _,
                last: _,
                pass: _,
                assigned: _,
            } => {
                self.mark_object(*attacker);
                self.mark_object(*blocker);
            }
            Decision::OrderReplacement {
                player: _,
                candidates,
            } => {
                for candidate in candidates {
                    self.mark_object(candidate.source());
                }
            }
            Decision::EnterAsCopy {
                player: _,
                object_id,
                candidates,
            } => {
                self.mark_object(*object_id);
                if !is_current {
                    for &id in candidates {
                        self.mark_permanent(id);
                    }
                }
            }
            Decision::ChooseCreatureType {
                player: _,
                object_id,
                scope: _,
            } => self.mark_object(*object_id),
            Decision::NameCard { .. }
            | Decision::ChoosePaymentX { .. }
            | Decision::SearchNamed { .. }
            | Decision::DistributeCounters { .. }
            | Decision::DiscardFromRevealedHand { .. }
            | Decision::Modal { .. }
            | Decision::Learn { .. }
            | Decision::Discard { .. } => {}
            Decision::Payment(progress) => self.walk_payment_progress(progress),
            Decision::ManaSpend {
                player: _,
                cost: _,
                spent,
                completion,
            } => {
                spent.visit_referenced_objects(|obj| self.mark_object(obj));
                match completion.as_ref() {
                    SpendCompletion::Announced(payment) => self.walk_payment(payment),
                    SpendCompletion::Upfront(inner) => self.walk_decision(inner, false),
                    SpendCompletion::Resolution(progress) => self.walk_payment_progress(progress),
                    SpendCompletion::Foretell(card) => self.mark_card(*card),
                }
            }
            Decision::ManaUpfront {
                player: _,
                permanent,
                ability_index: _,
                tap_remaining: _,
                resume,
            } => {
                self.mark_permanent(*permanent);
                if let Some(resume) = resume.as_deref() {
                    self.walk_decision(resume, false);
                }
            }
            Decision::ManaChoice {
                player: _,
                choices: _,
                production,
                resume,
            } => {
                match production {
                    ManaProduction::Output { source, tapped: _ } => self.mark_opt_object(*source),
                    ManaProduction::Bonus {
                        source,
                        tapped,
                        produced: _,
                        pending,
                    } => {
                        self.mark_object(*source);
                        self.mark_object(*tapped);
                        for PendingManaBonus { source, output: _ } in pending {
                            self.mark_object(*source);
                        }
                    }
                }
                if let Some(resume) = resume.as_deref() {
                    self.walk_decision(resume, false);
                }
            }
            Decision::ChooseObject {
                player: _,
                predicate: _,
                excluded,
                optional: _,
                effect: _,
            } => self.mark_opt_permanent(*excluded),
            Decision::SelectCards {
                origin: _,
                player: _,
                from: _,
                candidates,
                selected,
                min: _,
                max: _,
                effects: _,
                if_declined: _,
            } => {
                if !is_current {
                    for &object in candidates {
                        self.mark_object(object);
                    }
                }
                for &object in selected {
                    self.mark_object(object);
                }
            }
            Decision::Scry {
                player: _,
                looked,
                remaining,
                top,
                bottom,
            } => {
                for &object in looked.iter().chain(remaining).chain(top).chain(bottom) {
                    self.mark_object(object);
                }
            }
            Decision::ManifestDread { player: _, looked } => {
                for &object in looked {
                    self.mark_object(object);
                }
            }
            Decision::Surveil {
                player: _,
                looked,
                graveyard,
            } => {
                for &object in looked.iter().chain(graveyard) {
                    self.mark_object(object);
                }
            }
            Decision::LookAndSelect {
                random_order: _,
                reveal: _,
                to_battlefield: _,
                player: _,
                looked,
                predicate: _,
                selected: _,
                min_select: _,
                max_select: _,
            } => {
                for &object in looked {
                    self.mark_object(object);
                }
            }
            Decision::OrderLibrary {
                player: _,
                object_ids,
                next: _,
                end: _,
            } => {
                for &object in object_ids {
                    self.mark_object(object);
                }
            }
            Decision::SearchDistinctNames {
                player: _,
                predicate: _,
                remaining: _,
                selected,
                end: _,
            } => {
                for &object in selected {
                    self.mark_object(object);
                }
            }
            Decision::OpponentSplitsRevealed {
                player: _,
                owner: _,
                revealed,
                remaining: _,
            } => {
                for &object in revealed {
                    self.mark_object(object);
                }
            }
            Decision::CastDuringResolution {
                player: _,
                offer,
                paid: _,
                spends,
            } => {
                match offer {
                    ResolutionCastOffer::Graveyard { .. } | ResolutionCastOffer::Hand { .. } => {}
                    ResolutionCastOffer::Exiled { object, below: _ }
                    | ResolutionCastOffer::LibraryTop { object } => self.mark_object(*object),
                    ResolutionCastOffer::ExiledWith {
                        source,
                        predicate: _,
                    } => self.mark_object(*source),
                    ResolutionCastOffer::AmongExiled { objects } => {
                        if !is_current {
                            for &object in objects {
                                self.mark_object(object);
                            }
                        }
                    }
                    ResolutionCastOffer::Cascade {
                        object,
                        exiled,
                        below: _,
                    } => {
                        self.mark_object(*object);
                        for &item in exiled {
                            self.mark_object(item);
                        }
                    }
                }
                if let Some((object, _, _)) = spends {
                    self.mark_object(*object);
                }
            }
            Decision::ExileLinked {
                player: _,
                owner: _,
                zone: _,
                predicate: _,
                up_to: _,
                grant: _,
                grantee: _,
                source,
            } => self.mark_opt_object(*source),
            Decision::KeepCreatures {
                player: _,
                max_total_power: _,
                sacrifice: _,
                kept,
            } => {
                for &id in kept {
                    self.mark_permanent(id);
                }
            }
            Decision::Retarget {
                player: _,
                object,
                slot: _,
                may_keep: _,
            } => self.mark_target(*object),
            Decision::ChooseToExile {
                player: _,
                controller: _,
                predicate: _,
                chosen,
            } => {
                for &id in chosen {
                    self.mark_permanent(id);
                }
            }
            Decision::LegendRule {
                player: _,
                groups,
                keep,
            } => {
                let start = usize::from(is_current);
                for group in groups.iter().skip(start) {
                    for &object in group {
                        self.mark_object(object);
                    }
                }
                for &object in keep {
                    self.mark_object(object);
                }
            }
        }
    }
}

/// Settled permanent `spell_entry` summary: only whether it was paid, whether
/// it was kicked, and its announced X are read after the permanent settles.
fn settled_spell_entry_key(entry: Option<&SpellEntry>) -> (bool, bool, u8) {
    match entry {
        None => (false, false, 0),
        Some(SpellEntry { kicked, x, payment }) => (payment.is_some(), *kicked, *x),
    }
}

/// Cheap, pre-layer permanent record comparison with exhaustive destructuring.
/// Summoning sickness, `entry_turn` and `timestamp` are checked after layered
/// characteristics are known.
fn same_permanent_record_cheap(left: &Permanent, right: &Permanent) -> bool {
    let Permanent {
        transforms,
        manifested,
        id: _,     // unique public id
        object: _, // unique object id (compared via Object::same_object_identity)
        controller,
        control_differs_from_entry,
        tapped,
        summoning_sick: _, // checked with layered characteristics
        damage,
        deathtouch_damage,
        dealt_damage_this_turn,
        entry_turn: _, // checked with layered characteristics (power_up)
        counters,
        keyword_counter_timestamps,
        timestamp: _, // checked via layered static_abilities
        attacking,
        spell_entry,
        attached_to,
        spent_abilities,
        activated_this_turn,
    } = left;
    let Permanent {
        transforms: r_transforms,
        manifested: r_manifested,
        id: _,
        object: _,
        controller: r_controller,
        control_differs_from_entry: r_control_differs_from_entry,
        tapped: r_tapped,
        summoning_sick: _,
        damage: r_damage,
        deathtouch_damage: r_deathtouch,
        dealt_damage_this_turn: r_dealt,
        entry_turn: _,
        counters: r_counters,
        keyword_counter_timestamps: r_kw_ts,
        timestamp: _,
        attacking: r_attacking,
        spell_entry: r_spell_entry,
        attached_to: r_attached_to,
        spent_abilities: r_spent,
        activated_this_turn: r_activated,
    } = right;

    // Keyword counters carry timestamps that order layer 6 effects (CR 613.7c).
    if !keyword_counter_timestamps.is_empty() || !r_kw_ts.is_empty() {
        return false;
    }

    transforms == r_transforms
        && manifested == r_manifested
        && controller == r_controller
        && control_differs_from_entry == r_control_differs_from_entry
        && tapped == r_tapped
        && damage == r_damage
        && deathtouch_damage == r_deathtouch
        && dealt_damage_this_turn == r_dealt
        && counters == r_counters
        && attacking == r_attacking
        && settled_spell_entry_key(spell_entry.as_deref())
            == settled_spell_entry_key(r_spell_entry.as_deref())
        && attached_to == r_attached_to
        && spent_abilities == r_spent
        && activated_this_turn == r_activated
}

pub(crate) fn can_become_creature(view: &Characteristics) -> bool {
    view.activated_abilities
        .iter()
        .any(|ability| ability.effects.iter().any(Effect::may_become_creature))
        || view.triggered_abilities.iter().any(|trigger| {
            trigger
                .ability
                .effects()
                .iter()
                .any(Effect::may_become_creature)
        })
        || view.static_abilities.iter().any(|ability| match ability {
            StaticAbility::Continuous(continuous) => continuous
                .modifications
                .iter()
                .any(Modification::grants_creature_type),
            StaticAbility::TypeGrant(grant) => grant.types.contains(&CardType::Creature),
            _ => false,
        })
}

/// Cheap bucket key for permanent actions.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PermanentBucket {
    def: CardDefId,
    owner: PlayerId,
    kind: ObjectKind,
    controller: PlayerId,
    tapped: bool,
    attacking: bool,
}

/// Cheap bucket key for non-battlefield zone cards and outside-game sideboard
/// cards (`zone == None`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct CardBucket {
    zone: Option<ZoneType>,
    def: CardDefId,
    owner: PlayerId,
    kind: ObjectKind,
}

/// Cheap bucket key for foldable actions.
/// Two actions can fold only if they share a `FoldKey`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FoldKey {
    TapForMana {
        player: PlayerId,
        perm: PermanentBucket,
    },
    ActivateAbility {
        player: PlayerId,
        ability_index: usize,
        perm: PermanentBucket,
    },
    WaterbendTap {
        player: PlayerId,
        perm: PermanentBucket,
    },
    PlayLand {
        player: PlayerId,
        profile: SpellProfile,
        route: Option<PlayRoute>,
        card: CardBucket,
    },
    CastSpell {
        player: PlayerId,
        profile: SpellProfile,
        route: Option<PlayRoute>,
        card: CardBucket,
    },
    ActivateCardAbility {
        player: PlayerId,
        ability_index: usize,
        card: CardBucket,
    },
    Foretell {
        player: PlayerId,
        card: CardBucket,
    },
    ChooseTargetPermanent {
        player: PlayerId,
        perm: PermanentBucket,
    },
    ChooseTargetGraveyard {
        player: PlayerId,
        card: CardBucket,
    },
    SelectCardPermanent {
        player: PlayerId,
        perm: PermanentBucket,
    },
    SelectCard {
        player: PlayerId,
        card: CardBucket,
    },
    LearnDiscard {
        player: PlayerId,
        card: CardBucket,
    },
    LearnTakeLesson {
        player: PlayerId,
        card: CardBucket,
    },
    DeclareBlocker {
        player: PlayerId,
        blocker: PermanentId,
        attacker: PermanentBucket,
    },
}

impl FoldKey {
    #[inline]
    fn is_permanent(self) -> bool {
        matches!(
            self,
            Self::TapForMana { .. }
                | Self::ActivateAbility { .. }
                | Self::WaterbendTap { .. }
                | Self::ChooseTargetPermanent { .. }
                | Self::SelectCardPermanent { .. }
                | Self::DeclareBlocker { .. }
        )
    }
}

/// Whether two actions differ at most in the object/permanent they name,
/// sharing the same action variant, player, profile, route, ability index,
/// and target kind.
fn same_action_shell(left: &Action, right: &Action) -> bool {
    match (left, right) {
        (
            Action::PlayLand {
                profile: lp,
                player: lpl,
                route: lr,
                ..
            },
            Action::PlayLand {
                profile: rp,
                player: rpl,
                route: rr,
                ..
            },
        )
        | (
            Action::CastSpell {
                profile: lp,
                player: lpl,
                route: lr,
                ..
            },
            Action::CastSpell {
                profile: rp,
                player: rpl,
                route: rr,
                ..
            },
        ) => lp == rp && lpl == rpl && lr == rr,
        (
            Action::ActivateAbility {
                player: lpl,
                ability_index: li,
                ..
            },
            Action::ActivateAbility {
                player: rpl,
                ability_index: ri,
                ..
            },
        )
        | (
            Action::ActivateCardAbility {
                player: lpl,
                ability_index: li,
                ..
            },
            Action::ActivateCardAbility {
                player: rpl,
                ability_index: ri,
                ..
            },
        ) => lpl == rpl && li == ri,
        (Action::Foretell { player: lpl, .. }, Action::Foretell { player: rpl, .. })
        | (Action::TurnFaceUp { player: lpl, .. }, Action::TurnFaceUp { player: rpl, .. })
        | (Action::TapForMana { player: lpl, .. }, Action::TapForMana { player: rpl, .. })
        | (Action::WaterbendTap { player: lpl, .. }, Action::WaterbendTap { player: rpl, .. })
        | (Action::SelectCard { player: lpl, .. }, Action::SelectCard { player: rpl, .. })
        | (Action::LearnDiscard { player: lpl, .. }, Action::LearnDiscard { player: rpl, .. })
        | (
            Action::LearnTakeLesson { player: lpl, .. },
            Action::LearnTakeLesson { player: rpl, .. },
        ) => lpl == rpl,
        (
            Action::ChooseTarget {
                player: lpl,
                target: lt,
            },
            Action::ChooseTarget {
                player: rpl,
                target: rt,
            },
        ) => {
            lpl == rpl
                && matches!(
                    (lt, rt),
                    (Target::Permanent(_), Target::Permanent(_))
                        | (Target::GraveyardCard(_), Target::GraveyardCard(_))
                )
        }
        (
            Action::DeclareBlocker {
                player: lpl,
                blocker: lb,
                attacker: Some(_),
            },
            Action::DeclareBlocker {
                player: rpl,
                blocker: rb,
                attacker: Some(_),
            },
        ) => lpl == rpl && lb == rb,
        _ => false,
    }
}

impl Game {
    /// Folded copies for the currently published action space, as
    /// `(surviving_action_index, folded_object)` pairs.
    pub fn action_aliases(&self) -> &[(usize, ObjectId)] {
        self.action_aliases.as_deref().unwrap_or(&[])
    }

    /// Whether two objects in the same zone (or both outside the game) are
    /// game-indistinguishable: identical state, neither face down, no
    /// layer-ordering timestamp, and no live rules reference to either.
    pub fn interchangeable(&self, a: ObjectId, b: ObjectId) -> bool {
        if a == b {
            return true;
        }
        self.all_interchangeable(&[a, b])
    }

    /// Whether every object in `objects` belongs to one interchangeable class,
    /// running the reference census and `board_scan` at most once.
    pub(crate) fn all_interchangeable(&self, objects: &[ObjectId]) -> bool {
        let Some((&first, rest)) = objects.split_first() else {
            return true;
        };
        if rest.is_empty() {
            return true;
        }
        let zone = self.state.zones.zone_of(first);
        if !rest
            .iter()
            .all(|&obj| self.state.zones.zone_of(obj) == zone)
        {
            return false;
        }
        if !self.same_object_state_cheap(zone, first, rest) {
            return false;
        }
        let census = ReferenceCensus::run(self, objects);
        if (0..objects.len()).any(|i| census.is_named(i)) {
            return false;
        }
        if zone == Some(ZoneType::Battlefield) {
            let _scan = self.board_scan();
            let first_perm = self.state.permanent_of(first);
            let Some(first_perm) = first_perm else {
                return false;
            };
            let left_view = self.characteristics(first_perm);
            if !left_view.static_abilities.is_empty() {
                return false;
            }
            for &other in rest {
                let Some(other_perm) = self.state.permanent_of(other) else {
                    return false;
                };
                if !self.same_layered_pair(first_perm, &left_view, other_perm) {
                    return false;
                }
            }
        }
        true
    }

    /// Whether two battlefield permanents are currently game-indistinguishable.
    pub(crate) fn interchangeable_permanents(&self, a: PermanentId, b: PermanentId) -> bool {
        if a == b {
            return true;
        }
        let (Some(oa), Some(ob)) = (self.permanent_object(a), self.permanent_object(b)) else {
            return false;
        };
        self.interchangeable(oa, ob)
    }

    /// Whether an interchangeable creature declined to attack earlier in the
    /// current declaration.
    pub(crate) fn declined_interchangeable_attacker(&self, attacker: PermanentId) -> bool {
        let Some(combat) = self.state.combat.as_deref() else {
            return false;
        };
        self.declined_interchangeable_permanent(attacker, &combat.declined_attackers)
    }

    /// Whether an interchangeable creature declined to block earlier in the
    /// current declaration.
    pub(crate) fn declined_interchangeable_blocker(&self, blocker: PermanentId) -> bool {
        let Some(combat) = self.state.combat.as_deref() else {
            return false;
        };
        self.declined_interchangeable_permanent(blocker, &combat.declined_blockers)
    }

    fn declined_interchangeable_permanent(
        &self,
        candidate: PermanentId,
        declined: &[PermanentId],
    ) -> bool {
        if declined.is_empty() {
            return false;
        }
        let Some((cand_bucket, cand_object)) = self.permanent_bucket(candidate) else {
            return false;
        };
        let Some(cand_perm) = self.state.permanents[candidate].as_ref() else {
            return false;
        };
        let cand_obj = &self.state[cand_object];

        let mut matches: Vec<(PermanentId, ObjectId)> = Vec::new();
        for &other in declined {
            if other == candidate {
                continue;
            }
            let Some((other_bucket, other_object)) = self.permanent_bucket(other) else {
                continue;
            };
            if other_bucket != cand_bucket {
                continue;
            }
            let Some(other_perm) = self.state.permanents[other].as_ref() else {
                continue;
            };
            if !cand_obj.same_object_identity(&self.state[other_object]) {
                continue;
            }
            if !self.same_permanent_state_cheap(candidate, cand_perm, other, other_perm) {
                continue;
            }
            matches.push((other, other_object));
        }
        if matches.is_empty() {
            return false;
        }

        let mut watch = Vec::with_capacity(1 + matches.len());
        watch.push(cand_object);
        for &(_, obj) in &matches {
            watch.push(obj);
        }
        let census = ReferenceCensus::run(self, &watch);
        if census.is_named(0) {
            return false;
        }

        let _scan = self.board_scan();
        let cand_view = self.characteristics(candidate);
        if !cand_view.static_abilities.is_empty() {
            return false;
        }
        for (i, &(other, _)) in matches.iter().enumerate() {
            if census.is_named(i + 1) {
                continue;
            }
            if self.same_layered_pair(candidate, &cand_view, other) {
                return true;
            }
        }
        false
    }

    /// Compare two battlefield permanents on their cheap permanent record,
    /// `attacks_this_turn` count, and combat role (same attacking status and
    /// neither already blocked or assigned blockers).
    fn same_permanent_state_cheap(
        &self,
        left_id: PermanentId,
        left: &Permanent,
        right_id: PermanentId,
        right: &Permanent,
    ) -> bool {
        if !same_permanent_record_cheap(left, right) {
            return false;
        }
        if !self.state.turn.attacks_this_turn.is_empty()
            && self.state.turn.attack_count(left.object)
                != self.state.turn.attack_count(right.object)
        {
            return false;
        }
        self.same_combat_role(left_id, right_id)
    }

    fn same_combat_role(&self, left: PermanentId, right: PermanentId) -> bool {
        let Some(combat) = self.state.combat.as_deref() else {
            return true;
        };
        if combat.attackers.contains(&left) != combat.attackers.contains(&right) {
            return false;
        }
        let left_blocked = combat.blocked.contains(&left)
            || combat
                .attacker_to_blockers
                .get(&left)
                .is_some_and(|blockers| !blockers.is_empty());
        let right_blocked = combat.blocked.contains(&right)
            || combat
                .attacker_to_blockers
                .get(&right)
                .is_some_and(|blockers| !blockers.is_empty());
        !left_blocked && !right_blocked
    }

    /// Cheap (pre-census, pre-layer) object and permanent state check.
    fn same_object_state_cheap(
        &self,
        zone: Option<ZoneType>,
        first: ObjectId,
        rest: &[ObjectId],
    ) -> bool {
        let first_obj = match zone {
            Some(_) => match self.state.object(first) {
                Ok(obj) => obj,
                Err(_) => return false,
            },
            None => {
                if !self.state.zones.is_latest(first) {
                    return false;
                }
                self.state.entry_object(first)
            }
        };
        if first_obj.is_face_down() {
            return false;
        }
        let first_perm = if zone == Some(ZoneType::Battlefield) {
            let Some(id) = self.state.permanent_of(first) else {
                return false;
            };
            self.state.permanents[id].as_ref().map(|p| (id, p))
        } else {
            None
        };
        for &other in rest {
            let other_obj = match zone {
                Some(_) => match self.state.object(other) {
                    Ok(obj) => obj,
                    Err(_) => return false,
                },
                None => {
                    if !self.state.zones.is_latest(other) {
                        return false;
                    }
                    self.state.entry_object(other)
                }
            };
            if !first_obj.same_object_identity(other_obj) {
                return false;
            }
            if let Some((left_id, left)) = first_perm {
                let Some(other_id) = self.state.permanent_of(other) else {
                    return false;
                };
                let Some(right) = self.state.permanents[other_id].as_ref() else {
                    return false;
                };
                if !self.same_permanent_state_cheap(left_id, left, other_id, right) {
                    return false;
                }
            }
        }
        true
    }

    /// Layered characteristics, timestamp/static rule, summoning sickness and
    /// `power_up` turn-of-entry check for two battlefield permanents that
    /// already passed `same_object_identity` and `same_permanent_record_cheap`.
    /// Caller holds a `board_scan()`.
    fn same_layered_pair(
        &self,
        left_id: PermanentId,
        left_view: &std::sync::Arc<Characteristics>,
        right_id: PermanentId,
    ) -> bool {
        let right_view = self.characteristics(right_id);
        // Permanents with any layered (printed or granted) static ability have
        // layer-ordering timestamps (CR 613.7) and never fold.
        if !left_view.static_abilities.is_empty() || !right_view.static_abilities.is_empty() {
            return false;
        }
        if !std::sync::Arc::ptr_eq(left_view, &right_view) && left_view != &right_view {
            return false;
        }
        let left = self.state.permanents[left_id].as_ref().unwrap();
        let right = self.state.permanents[right_id].as_ref().unwrap();
        // Summoning sickness matters for creatures (CR 302.6), non-lands
        // (Vehicles, Spacecraft), and self-animating lands. Plain lands played
        // this turn are interchangeable with earlier copies.
        if left.summoning_sick != right.summoning_sick
            && (left_view.is_creature() || !left_view.is_land() || can_become_creature(left_view))
        {
            return false;
        }
        let turn = self.state.turn.turn_number;
        if left.entered_this_turn(turn) != right.entered_this_turn(turn)
            && left_view
                .activated_abilities
                .iter()
                .any(|ability| ability.power_up)
        {
            return false;
        }
        true
    }

    /// Find `wanted` in `space`, falling back to `self.action_aliases()` when
    /// folding replaced `wanted`'s object with an interchangeable representative.
    pub(crate) fn resolve_action_index(
        &self,
        space: &ActionSpace,
        wanted: &Action,
    ) -> Option<usize> {
        if let Some(index) = space.actions.iter().position(|legal| legal == wanted) {
            return Some(index);
        }
        let wanted_object = self.action_fold_object(wanted)?;
        for &(surviving_idx, alias_object) in self.action_aliases() {
            if alias_object != wanted_object {
                continue;
            }
            if let Some(surviving) = space.actions.get(surviving_idx) {
                if same_action_shell(surviving, wanted) {
                    return Some(surviving_idx);
                }
            }
        }
        None
    }

    fn action_fold_object(&self, action: &Action) -> Option<ObjectId> {
        match *action {
            Action::PlayLand { object, .. }
            | Action::CastSpell { object, .. }
            | Action::ActivateCardAbility { object, .. }
            | Action::Foretell { object, .. }
            | Action::TurnFaceUp { object, .. }
            | Action::SelectCard { object, .. }
            | Action::LearnDiscard { object, .. }
            | Action::LearnTakeLesson { object, .. } => Some(object),
            Action::TapForMana { permanent, .. }
            | Action::ActivateAbility { permanent, .. }
            | Action::WaterbendTap { permanent, .. } => self.permanent_object(permanent),
            Action::ChooseTarget {
                target: Target::Permanent(permanent),
                ..
            } => self.permanent_object(permanent),
            Action::ChooseTarget {
                target: Target::GraveyardCard(object),
                ..
            } => Some(object),
            Action::DeclareBlocker {
                attacker: Some(attacker),
                ..
            } => self.permanent_object(attacker),
            _ => None,
        }
    }

    /// Whether the active decision allows folding `Action::SelectCard`.
    /// `OrderLibrary` and `DistributeCounters` are never folded even when
    /// their candidates would otherwise look identical.
    fn select_card_decision_allows_fold(&self) -> bool {
        if self.state.opening().is_some() || self.pending_choice.is_some() {
            return true;
        }
        let Some(suspended) = &self.state.suspended_decision else {
            return true;
        };
        !matches!(
            suspended.decision,
            Decision::OrderLibrary { .. } | Decision::DistributeCounters { .. }
        )
    }

    fn permanent_bucket(&self, permanent: PermanentId) -> Option<(PermanentBucket, ObjectId)> {
        let perm = self.state.permanents.get(permanent.0)?.as_ref()?;
        let obj = self.state.object(perm.object).ok()?;
        if obj.is_face_down() || !perm.keyword_counter_timestamps.is_empty() {
            return None;
        }
        Some((
            PermanentBucket {
                def: obj.definition_id,
                owner: obj.owner,
                kind: obj.kind(),
                controller: perm.controller,
                tapped: perm.tapped,
                attacking: perm.attacking,
            },
            perm.object,
        ))
    }

    fn zone_card_bucket(&self, object: ObjectId, expected: Option<ZoneType>) -> Option<CardBucket> {
        let zone = self.state.zones.zone_of(object)?;
        if !matches!(zone, ZoneType::Hand | ZoneType::Graveyard | ZoneType::Exile) {
            return None;
        }
        if expected.is_some_and(|want| want != zone) {
            return None;
        }
        let obj = self.state.object(object).ok()?;
        if obj.is_face_down() {
            return None;
        }
        Some(CardBucket {
            zone: Some(zone),
            def: obj.definition_id,
            owner: obj.owner,
            kind: obj.kind(),
        })
    }

    fn sideboard_card_bucket(&self, object: ObjectId) -> Option<CardBucket> {
        if self.state.zones.zone_of(object).is_some() || !self.state.zones.is_latest(object) {
            return None;
        }
        let card = self.state.card_id(object)?;
        let owner = self.state.cards[card].owner;
        if !self.state.players[owner.0].sideboard.contains(&card) {
            return None;
        }
        let obj = self.state.entry_object(object);
        if obj.is_face_down() {
            return None;
        }
        Some(CardBucket {
            zone: None,
            def: obj.definition_id,
            owner: obj.owner,
            kind: obj.kind(),
        })
    }

    /// Extract the cheap bucket key and object id for a foldable action.
    /// Face-down objects, library/stack candidates, and disallowed decisions
    /// return `None` immediately (never folded).
    fn fold_candidate(
        &self,
        action: &Action,
        allow_select_card: bool,
    ) -> Option<(FoldKey, ObjectId)> {
        match *action {
            Action::TapForMana { player, permanent } => {
                let (perm, object) = self.permanent_bucket(permanent)?;
                Some((FoldKey::TapForMana { player, perm }, object))
            }
            Action::ActivateAbility {
                player,
                permanent,
                ability_index,
            } => {
                let (perm, object) = self.permanent_bucket(permanent)?;
                Some((
                    FoldKey::ActivateAbility {
                        player,
                        ability_index,
                        perm,
                    },
                    object,
                ))
            }
            Action::WaterbendTap { player, permanent } => {
                let (perm, object) = self.permanent_bucket(permanent)?;
                Some((FoldKey::WaterbendTap { player, perm }, object))
            }
            Action::PlayLand {
                profile,
                player,
                object,
                route,
            } => {
                let card = self.zone_card_bucket(object, None)?;
                Some((
                    FoldKey::PlayLand {
                        player,
                        profile,
                        route,
                        card,
                    },
                    object,
                ))
            }
            Action::CastSpell {
                profile,
                player,
                object,
                route,
            } => {
                let card = self.zone_card_bucket(object, None)?;
                Some((
                    FoldKey::CastSpell {
                        player,
                        profile,
                        route,
                        card,
                    },
                    object,
                ))
            }
            Action::ActivateCardAbility {
                player,
                object,
                ability_index,
            } => {
                let card = self.zone_card_bucket(object, None)?;
                Some((
                    FoldKey::ActivateCardAbility {
                        player,
                        ability_index,
                        card,
                    },
                    object,
                ))
            }
            Action::Foretell { player, object } => {
                let card = self.zone_card_bucket(object, Some(ZoneType::Hand))?;
                Some((FoldKey::Foretell { player, card }, object))
            }
            Action::ChooseTarget {
                player,
                target: Target::Permanent(permanent),
            } => {
                let (perm, object) = self.permanent_bucket(permanent)?;
                Some((FoldKey::ChooseTargetPermanent { player, perm }, object))
            }
            Action::ChooseTarget {
                player,
                target: Target::GraveyardCard(object),
            } => {
                let card = self.zone_card_bucket(object, Some(ZoneType::Graveyard))?;
                Some((FoldKey::ChooseTargetGraveyard { player, card }, object))
            }
            Action::SelectCard { player, object } if allow_select_card => {
                match self.state.zones.zone_of(object) {
                    Some(ZoneType::Battlefield) => {
                        let permanent = self.state.permanent_of(object)?;
                        let (perm, obj) = self.permanent_bucket(permanent)?;
                        Some((FoldKey::SelectCardPermanent { player, perm }, obj))
                    }
                    Some(ZoneType::Hand | ZoneType::Graveyard | ZoneType::Exile) => {
                        let card = self.zone_card_bucket(object, None)?;
                        Some((FoldKey::SelectCard { player, card }, object))
                    }
                    None => {
                        let card = self.sideboard_card_bucket(object)?;
                        Some((FoldKey::SelectCard { player, card }, object))
                    }
                    Some(ZoneType::Library | ZoneType::Stack | ZoneType::Command) => None,
                }
            }
            Action::LearnDiscard { player, object } => {
                let card = self.zone_card_bucket(object, Some(ZoneType::Hand))?;
                Some((FoldKey::LearnDiscard { player, card }, object))
            }
            Action::LearnTakeLesson { player, object } => {
                let card = self.sideboard_card_bucket(object)?;
                Some((FoldKey::LearnTakeLesson { player, card }, object))
            }
            Action::DeclareBlocker {
                player,
                blocker,
                attacker: Some(attacker_id),
            } => {
                let (attacker, object) = self.permanent_bucket(attacker_id)?;
                Some((
                    FoldKey::DeclareBlocker {
                        player,
                        blocker,
                        attacker,
                    },
                    object,
                ))
            }
            // Face-down TurnFaceUp, Library/Stack targets, ChooseMode,
            // ScryCard, ChooseTrigger, PassPriority, PayCost, Decline, and
            // DeclareAttacker / DeclareBlocker(None) never fold here.
            _ => None,
        }
    }

    /// Fold actions that differ only in which of several game-indistinguishable
    /// objects they name. The first action of each class in emission order
    /// survives; folded copies are returned as `(surviving_index, object)`
    /// alias pairs.
    ///
    /// Fast path: zero heap allocations when no two foldable actions share a
    /// cheap bucket key.
    pub(crate) fn fold_action_space(&self, mut space: ActionSpace) -> (ActionSpace, ActionAliases) {
        if space.actions.len() < 2 {
            return (space, None);
        }

        let allow_select_card = self.select_card_decision_allows_fold();

        // Stack-buffered candidate scan: avoids any heap allocation in the
        // common case where no two foldable actions share a bucket key.
        let mut inline: [Option<(FoldKey, usize, ObjectId)>; 24] = [None; 24];
        let mut inline_len = 0usize;
        let mut spill: Vec<(FoldKey, usize, ObjectId)> = Vec::new();
        let mut has_dup = false;

        for (index, action) in space.actions.iter().enumerate() {
            let Some((key, object)) = self.fold_candidate(action, allow_select_card) else {
                continue;
            };
            if spill.is_empty() {
                if !has_dup
                    && inline[..inline_len]
                        .iter()
                        .flatten()
                        .any(|(k, _, _)| *k == key)
                {
                    has_dup = true;
                }
                if inline_len < inline.len() {
                    inline[inline_len] = Some((key, index, object));
                    inline_len += 1;
                } else {
                    spill.reserve(space.actions.len());
                    spill.extend(inline[..inline_len].iter().flatten().copied());
                    spill.push((key, index, object));
                }
            } else {
                if !has_dup && spill.iter().any(|(k, _, _)| *k == key) {
                    has_dup = true;
                }
                spill.push((key, index, object));
            }
        }

        if !has_dup {
            return (space, None);
        }

        let candidates: Vec<(FoldKey, usize, ObjectId)> = if spill.is_empty() {
            inline[..inline_len].iter().flatten().copied().collect()
        } else {
            spill
        };

        // Mark which candidate entries belong to a bucket of size >= 2 and
        // build a deduplicated watch list for the reference census.
        let mut in_dup_bucket = vec![false; candidates.len()];
        let mut watch: Vec<ObjectId> = Vec::with_capacity(candidates.len());
        let mut watch_slot: Vec<usize> = vec![usize::MAX; candidates.len()];
        let mut has_perm_dup = false;
        let mut watch_index = |object: ObjectId| -> usize {
            if let Some(pos) = watch.iter().position(|&w| w == object) {
                pos
            } else {
                let pos = watch.len();
                watch.push(object);
                pos
            }
        };
        for i in 0..candidates.len() {
            if in_dup_bucket[i] {
                continue;
            }
            let key = candidates[i].0;
            let mut found = false;
            for j in (i + 1)..candidates.len() {
                if candidates[j].0 == key {
                    if !found {
                        found = true;
                        has_perm_dup |= key.is_permanent();
                        in_dup_bucket[i] = true;
                        watch_slot[i] = watch_index(candidates[i].2);
                    }
                    in_dup_bucket[j] = true;
                    watch_slot[j] = watch_index(candidates[j].2);
                }
            }
        }

        let census = ReferenceCensus::run(self, &watch);
        if census.all_named() {
            return (space, None);
        }

        let _scan = has_perm_dup.then(|| self.board_scan());

        // Map original action index -> (representative original action index, folded object)
        // for actions that fold into an earlier representative.
        let mut folded_into: Vec<Option<(usize, ObjectId)>> = vec![None; space.actions.len()];
        let mut any_folded = false;

        // Track representatives per duplicate bucket.
        let mut perm_reps: Vec<(
            FoldKey,
            usize,
            ObjectId,
            PermanentId,
            std::sync::Arc<Characteristics>,
        )> = Vec::new();
        let mut card_reps: Vec<(FoldKey, usize, ObjectId)> = Vec::new();

        for (ci, &(key, orig_idx, object)) in candidates.iter().enumerate() {
            if !in_dup_bucket[ci] || census.is_named(watch_slot[ci]) {
                continue;
            }
            if key.is_permanent() {
                let Some(perm_id) = self.state.permanent_of(object) else {
                    continue;
                };
                let Some(perm) = self.state.permanents[perm_id].as_ref() else {
                    continue;
                };
                let obj = &self.state[object];

                let mut matched_rep = None;
                for &(rep_key, rep_orig_idx, rep_object, rep_perm_id, ref rep_view) in &perm_reps {
                    if rep_key != key {
                        continue;
                    }
                    let rep_obj = &self.state[rep_object];
                    if !rep_obj.same_object_identity(obj) {
                        continue;
                    }
                    let rep_perm = self.state.permanents[rep_perm_id].as_ref().unwrap();
                    if !self.same_permanent_state_cheap(rep_perm_id, rep_perm, perm_id, perm) {
                        continue;
                    }
                    if self.same_layered_pair(rep_perm_id, rep_view, perm_id) {
                        matched_rep = Some(rep_orig_idx);
                        break;
                    }
                }

                if let Some(rep_orig_idx) = matched_rep {
                    folded_into[orig_idx] = Some((rep_orig_idx, object));
                    any_folded = true;
                } else {
                    let view = self.characteristics(perm_id);
                    if view.static_abilities.is_empty() {
                        perm_reps.push((key, orig_idx, object, perm_id, view));
                    }
                }
            } else {
                let obj = self.state.entry_object(object);
                let mut matched_rep = None;
                for &(rep_key, rep_orig_idx, rep_object) in &card_reps {
                    if rep_key != key {
                        continue;
                    }
                    if self
                        .state
                        .entry_object(rep_object)
                        .same_object_identity(obj)
                    {
                        matched_rep = Some(rep_orig_idx);
                        break;
                    }
                }

                if let Some(rep_orig_idx) = matched_rep {
                    folded_into[orig_idx] = Some((rep_orig_idx, object));
                    any_folded = true;
                } else {
                    card_reps.push((key, orig_idx, object));
                }
            }
        }

        if !any_folded {
            return (space, None);
        }

        // Compact surviving actions in emission order and remap alias indices
        // to surviving action indices.
        let orig_len = space.actions.len();
        let mut new_index = vec![0usize; orig_len];
        let mut kept = Vec::with_capacity(orig_len);
        let mut aliases: Vec<(usize, ObjectId)> = Vec::new();
        for (orig_idx, action) in space.actions.into_iter().enumerate() {
            if let Some((rep_orig_idx, object)) = folded_into[orig_idx] {
                aliases.push((new_index[rep_orig_idx], object));
            } else {
                new_index[orig_idx] = kept.len();
                kept.push(action);
            }
        }
        space.actions = kept;
        (space, Some(aliases.into_boxed_slice()))
    }
}
