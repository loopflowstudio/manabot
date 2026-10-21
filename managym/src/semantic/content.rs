// content.rs
// Lower checked-in semantic IR into the live rules engine's immutable content pack.

use std::{
    collections::BTreeSet,
    sync::{Arc, OnceLock},
};

use serde_json::Value;

use crate::{
    cardsets::alpha::{CompiledSemanticManifest, ContentPack},
    state::{
        ability::{
            Ability, Effect, EffectValue, StaticCondition, TargetRequirement, TargetSpec,
            TriggerCondition, TriggerSubject,
        },
        card::{
            ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, PowerCda,
            SacrificeCost, SpellCostReduction, StaticPtBuff, StaticScope, TriggeredManaAbility,
        },
        mana::{Color, Colors, Mana, ManaCost},
        player::PlayerConfig,
        predicate::CardPredicate,
    },
};

use super::{
    array_field, object_field, opt_bool_field, opt_str_field, parse_condition, str_array_field,
    str_field, u64_field, Condition, Definition, IrError, Predicate, Program, SemanticPack, Step,
    ValueExpression,
};

#[derive(Clone)]
struct AuthoredRuntime {
    pack_key: String,
    content: Arc<ContentPack>,
    setups: Vec<PlayerConfig>,
}

impl AuthoredRuntime {
    fn compile(semantic: SemanticPack) -> Result<Self, IrError> {
        let content = Arc::new(semantic.compile_content_pack()?);
        let setups = semantic
            .decks
            .iter()
            .map(|deck| semantic.player_config(&deck.key, &deck.key))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pack_key: semantic.pack_key,
            content,
            setups,
        })
    }

    fn matches(&self, player_configs: &[PlayerConfig]) -> bool {
        if player_configs.len() != self.setups.len() {
            return false;
        }
        let mut unmatched = self.setups.iter().collect::<Vec<_>>();
        for config in player_configs {
            let Some(index) = unmatched.iter().position(|setup| {
                setup.decklist == config.decklist && setup.sideboard == config.sideboard
            }) else {
                return false;
            };
            unmatched.remove(index);
        }
        unmatched.is_empty()
    }
}

static AUTHORED_RUNTIME_CATALOG: OnceLock<Result<Vec<AuthoredRuntime>, IrError>> = OnceLock::new();

fn authored_runtime_catalog() -> Result<Vec<AuthoredRuntime>, IrError> {
    AUTHORED_RUNTIME_CATALOG
        .get_or_init(|| {
            [
                SemanticPack::two_deck()?,
                SemanticPack::jeong_increment()?,
                SemanticPack::elemental()?,
                SemanticPack::sacrifice()?,
                SemanticPack::temur()?,
                SemanticPack::complexity()?,
