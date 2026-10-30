"""One-off: Rust follow-ups for continuous-effect records (source name,
experience wire view, decision-capture definitions)."""
def sub(p,old,new,cnt=1):
    s=open(p).read()
    if old not in s: raise SystemExit(f"missing in {p}: {old[:70]}")
    s=s.replace(old,new,cnt); open(p,'w').write(s)
R='managym/src/'
# observation: source_name
sub(R+'agent/observation.rs',"""    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_registry_key: Option<i32>,
    /// Timestamp order""","""    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_registry_key: Option<i32>,
    /// The source's printed name: public, since the spell or ability resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_name: Option<String>,
    /// Timestamp order""")
sub(R+'agent/observation.rs',"""    fn new(effect: &ContinuousEffect) -> Self {
        let mut data = Self {
            modification: String::new(),
            duration: format!("{:?}", effect.duration),
            source_registry_key: effect.source.map(|id| id.0 as i32),""","""    fn new(game: &Game, effect: &ContinuousEffect) -> Self {
        let mut data = Self {
            modification: String::new(),
            duration: format!("{:?}", effect.duration),
            source_registry_key: effect.source.map(|id| id.0 as i32),
            source_name: effect
                .source
                .and_then(|id| game.state.content.definition(id))
                .map(|definition| definition.name.clone()),""")
sub(R+'agent/observation.rs',"""            .filter(|effect| Some(effect.affected) == affected)
            .map(Self::new)""","""            .filter(|effect| Some(effect.affected) == affected)
            .map(|effect| Self::new(game, effect))""")
# bindings
sub(R+'python/bindings.rs',"""    #[pyo3(get, set)]
    pub source_registry_key: Option<i32>,
    #[pyo3(get, set)]
    pub timestamp: i32,
}""","""    #[pyo3(get, set)]
    pub source_registry_key: Option<i32>,
    #[pyo3(get, set)]
    pub source_name: Option<String>,
    #[pyo3(get, set)]
    pub timestamp: i32,
}""")
sub(R+'python/bindings.rs',"""            source_registry_key: value.source_registry_key,
            timestamp: value.timestamp,""","""            source_registry_key: value.source_registry_key,
            source_name: value.source_name,
            timestamp: value.timestamp,""",2)
# experience wire types
sub(R+'experience.rs',"""#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyPermanentView {""","""/// One continuous-effect record on a permanent or player (CR 611, 613).
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousEffectView {
    /// The engine's `Modification` variant.
    pub modification: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toughness: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subtypes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// `EndOfTurn` or `Indefinite`.
    pub duration: String,
    /// The source's printed name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Timestamp order (CR 613.7).
    pub timestamp: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyPermanentView {""")
sub(R+'experience.rs',"""    /// Activated abilities (indices in its ability list) whose limit is spent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub used_abilities: Vec<u32>,
}""","""    /// Activated abilities (indices in its ability list) whose limit is spent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub used_abilities: Vec<u32>,
    /// Continuous effects on it, in timestamp order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ContinuousEffectView>,
}""")
sub(R+'experience.rs',"""    /// Unspent red mana stays until end of turn.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub keeps_red_mana: bool,
    pub hand: Vec<LegacyCardView>,""","""    /// Unspent red mana stays until end of turn.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub keeps_red_mana: bool,
    /// Continuous effects on this player, in timestamp order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ContinuousEffectView>,
    pub hand: Vec<LegacyCardView>,""")
# decision capture: bind effect source definitions
sub(R+'decision_capture.rs',"""        for frame in &agent.continuation {
            ids.extend(frame.source.definition);""","""        for effects in [&agent.agent.effects, &agent.opponent.effects]
            .into_iter()
            .chain(
                agent
                    .agent_permanents
                    .iter()
                    .chain(&agent.opponent_permanents)
                    .map(|permanent| &permanent.effects),
            )
        {
            ids.extend(
                effects
                    .iter()
                    .filter_map(|effect| effect.source_registry_key)
                    .map(|key| key as u32),
            );
        }
        for frame in &agent.continuation {
            ids.extend(frame.source.definition);""")
