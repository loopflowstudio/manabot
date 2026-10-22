"""Retain the canonical diagnostic without rewriting earlier failure receipts."""
import json
from pathlib import Path

root = Path('docs/rules/audit-2026-10')
run = Path('.runs/learnability/rules-targeting-w12-canonical')
result = json.loads((run / 'result.json').read_text())
assert result['status'] == 'passed' and result['ppo']['passed']
(root / 'learnability-w12-canonical.json').write_bytes((run / 'result.json').read_bytes())
p = root / 'unification-validation.json'
data = json.loads(p.read_text())
data['canonical_runtime_validation'] = {
    'draft_checkpoint': '202416a8',
    'source_note': 'Diagnostic started on fe76226d with the working draft subsequently committed as 202416a8; source_digest and native SHA identify the tested bytes. Later report/helper edits are not training changes.',
    'learnability_result': 'learnability-w12-canonical.json',
    'learnability_path': str(run),
    'collector_profile': json.loads(Path('scratch/collector-profile.json').read_text()),
    'collector_diagnosis': 'Two unchanged PPO updates completed in 9.047s including startup/evaluation. The old 384.3s run consumed 76.2 CPU seconds. Host contention/scheduling is plausible but not proved; no collector defect was reproduced and no production repair was made.',
    'focused_protocol': json.loads(Path('scratch/canonical-study-protocol.json').read_text()),
    'focused_protocol_result': '17 passed, including dependency-blocked play and selected-offer-independent ed2 plus command-bound erd1 checks.',
    'budget_note': 'Unchanged 480s total watchdog, 300s PPO deadline, 64 updates x 128 steps x 8 streams, 64 evaluation games per arm, capacity thresholds and PPO assertions.',
    'remaining_gate_ci': ['Study runtime semantic retry', 'Seven Study branch cases', 'Four monitor cases: random/explicit-matrix outcomes, checkpoint-admission failure persistence, saved dashboard'],
    'evidence_scope': 'Earlier bounded consumer/debug/conformance receipts remain historical; this adds current canonical protocol and full learnability evidence only. No full conformance or strength certification.'
}
p.write_text(json.dumps(data, indent=2) + '\n')
p = root / 'README.md'
s = p.read_text().replace('Acceptance remains incomplete: full learnability failed its PPO collector\ntraining deadline. Study semantic retry, Study branch and four monitor cases\nremain after bounded 120-second timeouts. Partial suites are not reported as\npasses. The [validation receipts](unification-validation.json) retain both the\nfailures and repaired fixture results; no training or timeout limits were raised.', 'The canonical full [learnability diagnostic](learnability-w12-canonical.json)\nnow passes in 99.6s, including all three capacity seeds and reward-driven PPO\nwith checkpoint reload (65,536 transitions; after 1.0 versus no-update 0.640625).\nThe focused canonical dependency-blocked play/Study protocol check passes all\n17 tests. Earlier debug, conformance and consumer counts above are retained\nhistorical receipts, not newly rerun suites after the packaging merge.\n\nAcceptance remains incomplete: Study semantic retry, Study branch and four\nmonitor cases remain with gate/CI after bounded 120-second timeouts. Partial\nsuites are not passes. The earlier collector deadline failure remains in the\n[validation receipts](unification-validation.json). A bounded two-update profile\ncompleted normally; no collector defect was reproduced. Host contention is\nplausible but unproved. No training budgets, timeouts or assertions changed.')
p.write_text(s)
p = root / 'unification.md'
s = p.read_text().replace('CI selection/required-check change, training/deck/experiment change, publication,', 'required-check weakening, training-recipe/deck/experiment change, publication,')
s += '\n## Canonical runtime validation\n\nThe full [w12 diagnostic](learnability-w12-canonical.json) passes in 99.6s\nwith all 144 capacity assertions and reward-driven production PPO/checkpoint\nreload. PPO collected 65,536 learner transitions; after/no-update scores are\n1.0/0.640625. The unchanged two-update profile completed in 9.047s; no collector\ndefect was reproduced. Earlier low CPU-to-wall utilization suggests contention\nbut cannot prove the old timeout cause. No production training code, budget or\nassertion changed. Canonical dependency-blocked play/Study protocol passes\n17 tests. Historical receipts remain unchanged; Study runtime semantic retry,\nseven Study branch cases and four monitor cases remain with gate/CI.\n'
p.write_text(s)
p = Path('scratch/plan.md')
s = p.read_text().replace('plus a successful full learnability diagnostic including reward-driven PPO.', 'while full canonical learnability is now verified below.')
s = s.replace('One full canonical diagnostic is in progress with unchanged 480s total/300s PPO\ndeadlines, fixed sample counts and assertions; no timed-out broad suite is replayed.', 'The one full canonical diagnostic passed in 99.6s (86.5 CPU seconds), with\n144 passing capacity assertions and PPO checkpoint reload: 65,536 transitions,\n1,541 games, after 1.0 versus no-update 0.640625. The 480s total/300s PPO\ndeadlines, sample counts and assertions are unchanged. No production training\nrepair was warranted. Full result is retained at\ndocs/rules/audit-2026-10/learnability-w12-canonical.json; the old failure remains.\nCanonical dependency-blocked play/Study protocol also passes all 17 tests in\n2.28s. No timed-out broad suite was replayed.')
s = s.replace('Full PPO and unfinished Study/monitor acceptance remain with gate/CI.', 'Full canonical PPO now passes; unfinished Study/monitor acceptance remains with gate/CI. Draft checkpoint: 202416a8; no publication or lifecycle change.')
p.write_text(s)
