"""QA driver: learning_signal registration, served and scalar."""
import sys, time
from pathlib import Path
from etude.eval.records import Code, Model
from etude.experiment import Inference
import managym
from experiments import learning_signal as ls
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.sim.execution import EvalExecution

ckpt = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); mode = sys.argv[3]
design = ls.ALLIES_LESSONS
pairing = design.pairings[0]
matchup = design.matchup(pairing)
model = Model.model_validate({"name": "qa-candidate", "checkpoint": {
    "key": {"address": ckpt.as_uri()},
    "artifact": {"uri": ckpt.as_uri(), "sha256": file_sha256(ckpt), "size_bytes": ckpt.stat().st_size}},
    "inference": Inference(source=ls._source())})
heur = Code(name="heuristic", source=ls._source(), config={"kind": "heuristic"})
cand, cpath = ls._registration("candidate", model, matchup, design.name)
opp, _ = ls._registration("opponent", heur, matchup, design.name)
fresh = runtime_fingerprints(match_config=matchup.match())
print("obs abi", cand.observation_abi_sha256[:12], "fresh", fresh["observation_abi_sha256"][:12], "artifact", cand.artifact_id[:30], cand.role, opp.role)
key = EvaluationKey(world=managym.WORLD_VERSION, content_suite=matchup.content_suite,
    viewer_boundary="acting-viewer", arena_version=design.name, rating_model_version="unrated",
    rating_prior_sha256=canonical_sha256({}), anchor_cohort_sha256=canonical_sha256(opp.model_dump(mode="json")),
    evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass")
req = EvaluationRequest(key=key, candidate=cand, opponent=opp, checkpoint_paths={"candidate": str(cpath)},
    deal_seeds=(pairing.deal,), authored_matchup=matchup, matchup_mode=design.matchup_mode,
    game_seconds=design.game_seconds, purpose="qa-local")
execution = EvalExecution(device="cpu", deadline_unix=time.time() + 900) if mode == "served" else None
t = time.time(); res = evaluate(req, out_dir=out, execution=execution)
print("seconds", round(time.time() - t, 1), "error", res.error)
for r in res.rows: print("leg", r.leg, "valid", r.valid, "score_a", r.score_a, "failure", r.failure)
