use managym::study::StudyDecisionIndex;
use schemars::schema_for;

fn main() {
    let schema = schema_for!(StudyDecisionIndex);
    println!(
        "{}",
        serde_json::to_string_pretty(&schema).expect("schema serializes")
    );
}
