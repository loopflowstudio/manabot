// scenario.rs
// State-injection helpers for constructing specific game positions.
//
// FOR TEST AND MEASUREMENT HARNESSES ONLY — these methods mutate zones and
// player state directly, bypassing the rules engine (no events, no triggers,
// no costs). They mirror the Scenario helpers in managym/tests/rules/
// helpers.rs so the Python competency harness (manabot/verify/competency.py)
// can construct mid-game positions and score agents against known-correct
// lines. Injection clears derived history: there is no authentic transition
// into an injected position. Never call these from gameplay code paths.

use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock},
};

use crate::{
    agent::action::{ActionSpace, ActionSpaceKind, AgentError},
    cards::names::{NameCandidate, NameVocabulary},
    flow::{decision::Decision, game::Game},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        permanent::Permanent,
        zone::ZoneType,
    },
};

/// The checked-in, provenance-bearing name fixture adds no playable definitions.
/// Shared by native, policy and Academy proofs; never installed by game setup.
pub(crate) fn naming_fixture_vocabulary() -> Arc<NameVocabulary> {
    #[derive(serde::Deserialize)]
    struct Entry {
        name: String,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        names: Vec<Entry>,
    }
    static NAMES: LazyLock<Arc<NameVocabulary>> = LazyLock::new(|| {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/naming_vocabulary.json"))
                .expect("checked-in naming fixture");
        let mut entries: BTreeMap<_, _> = fixture
            .names
            .into_iter()
            .map(|entry| (entry.name, None))
            .collect();
        for entry in crate::cards::content_pack::default_content_pack()
            .name_candidates()
            .iter()
        {
            entries.insert(entry.name.clone(), entry.definition);
        }
        NameVocabulary::new(
            entries
                .into_iter()
                .map(|(name, definition)| NameCandidate { name, definition })
                .collect(),
        )
    });
    Arc::clone(&NAMES)
}

impl Game {
    /// Replace only a pending naming decision with the validated large fixture.
    /// This is an injected root, not a legal transition or production vocabulary.
    pub fn scenario_naming_vocabulary(&mut self) -> Result<Arc<NameVocabulary>, AgentError> {
        let Some(Decision::NameCard { names, scope, .. }) = self
            .state
            .suspended_decision
            .as_mut()
            .map(|pending| &mut pending.decision)
        else {
            return Err(AgentError(
                "scenario_naming_vocabulary requires a naming decision".into(),
            ));
        };
        let fixture = naming_fixture_vocabulary();
        *names = Arc::clone(&fixture);
        *scope = fixture.root();
        self.policy_history = Default::default();
        self.publish_action_space(
            self.suspended_decision_action_space()
                .expect("naming decision"),
        );
        Ok(fixture)
    }

    /// Set a player's life total directly.
    pub fn scenario_set_life(&mut self, player: PlayerId, life: i32) {
        self.policy_history = Default::default();
        self.state.players[player.0].life = life;
    }

    /// Move every card in `player`'s hand to the bottom of their library.
    pub fn scenario_clear_hand(&mut self, player: PlayerId) {
        self.policy_history = Default::default();
        let hand: Vec<CardId> = self.state.zones.zone_cards(ZoneType::Hand, player).to_vec();
        for card in hand {
            self.state.zones.move_card(card, player, ZoneType::Library);
            // move_card pushes to the top (end); rotate to the bottom so the
            // cleared hand is not immediately drawn back.
            let library = self.state.zones.zone_cards_mut(ZoneType::Library, player);
            if let Some(top) = library.pop() {
                library.insert(0, top);
            }
        }
    }

    /// Move one card named `name`, owned by `player`, from their library
    /// (preferred) or graveyard into their hand.
    pub fn scenario_force_card_in_hand(
        &mut self,
        player: PlayerId,
        name: &str,
    ) -> Result<(), AgentError> {
        for zone in [ZoneType::Library, ZoneType::Graveyard] {
            let found = self
                .state
                .zones
                .zone_cards(zone, player)
                .iter()
                .copied()
                .find(|card| self.state.cards[*card].name == name);
            if let Some(card) = found {
                self.policy_history = Default::default();
                self.state.zones.move_card(card, player, ZoneType::Hand);
                return Ok(());
            }
        }
        Err(AgentError(format!(
            "scenario_force_card_in_hand: no '{name}' in library/graveyard of player {}",
            player.0
        )))
    }

    /// Put a card named `name`, owned by `player`, onto the battlefield as a
    /// new permanent (no ETB triggers fire). With `ready`, the permanent is
    /// untapped and not summoning-sick — a creature that has "survived a
    /// turn" and may attack or tap immediately.
    pub fn scenario_force_battlefield(
        &mut self,
        player: PlayerId,
        name: &str,
        ready: bool,
    ) -> Result<PermanentId, AgentError> {
        let found = [ZoneType::Library, ZoneType::Hand, ZoneType::Graveyard]
            .into_iter()
            .find_map(|zone| {
                self.state
                    .zones
                    .zone_cards(zone, player)
                    .iter()
                    .copied()
                    .find(|card| self.state.cards[*card].name == name)
            });
        let Some(card_id) = found else {
            return Err(AgentError(format!(
                "scenario_force_battlefield: no '{name}' outside battlefield for player {}",
                player.0
            )));
        };

        self.policy_history = Default::default();
        self.state
            .zones
            .move_card(card_id, player, ZoneType::Battlefield);

        let permanent_id = PermanentId(self.state.permanents.len());
        let mut permanent = Permanent::new(
            self.state.id_gen_mut().next_id(),
            card_id,
            &self.state.cards[card_id],
        );
        if ready {
            permanent.summoning_sick = false;
            permanent.tapped = false;
        }
        self.state.permanents.push(Some(permanent));
        if self.state.card_to_permanent.len() <= card_id.0 {
            self.state.card_to_permanent.resize(card_id.0 + 1, None);
        }
        self.state.card_to_permanent[card_id] = Some(permanent_id);
        self.invalidate_mana_cache(player);
        Ok(permanent_id)
    }

    /// Recompute the current priority action space after state injection.
    ///
    /// Injection leaves `current_action_space` stale (it was computed before
    /// hands/battlefield changed). Only priority spaces can be refreshed;
    /// call this once, after all injections, while the game is at a priority
    /// decision (e.g. immediately after reset, at the first main phase).
    pub fn scenario_refresh_priority(&mut self) -> Result<(), AgentError> {
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
        let Some(space) = self.current_action_space.as_ref() else {
            return Err(AgentError(
                "scenario_refresh_priority: no active action space".to_string(),
            ));
        };
        if space.kind != ActionSpaceKind::Priority {
            return Err(AgentError(format!(
                "scenario_refresh_priority: current action space is {:?}, not Priority",
                space.kind
            )));
        }
        let player = space
            .actions
            .first()
            .map(|action| action.player())
            .ok_or_else(|| {
                AgentError("scenario_refresh_priority: priority space has no player".to_string())
            })?;
        let actions = self.compute_player_actions(player);
        self.publish_action_space(ActionSpace {
            player: Some(player),
            kind: ActionSpaceKind::Priority,
            actions,
            focus: Vec::new(),
        });
        Ok(())
    }
}
