use managym::study::RecordedDecisionInput;
use schemars::schema_for;

fn main() {
    let schema = schema_for!(RecordedDecisionInput);
    println!(
        "{}",
        serde_json::to_string_pretty(&schema).expect("schema serializes")
    );
}
