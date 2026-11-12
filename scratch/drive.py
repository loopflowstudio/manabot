"""QA driver: frontier-shaped checkpoint vs heuristic, served and scalar."""
import json, sys, time
from pathlib import Path
from etude.eval.records import Code, Model
import managym
from experiments import tla_frontier as tf
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.sim.execution import EvalExecution

ckpt = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); mode = sys.argv[3]
regimes, protocol = tf.frozen()
regime = regimes[tf.SMALL]
pair = protocol.pairs[0]
matchup = tf.deck_matchup(regime, protocol, pair)
from manabot.infra.identity import file_sha256
model = Model.model_validate({"name": "qa-candidate", "checkpoint": {
    "key": {"address": ckpt.as_uri()},
    "artifact": {"uri": ckpt.as_uri(), "sha256": file_sha256(ckpt), "size_bytes": ckpt.stat().st_size}},
    "inference": tf.checkpoint_inference()})
heur = Code(name="heuristic", source=tf._source(), config=tf.HEURISTIC)
cand, cpath = tf._registration(model, "candidate", matchup, regime.observation)
opp, _ = tf._registration(heur, "opponent", matchup, regime.observation)
from manabot.infra.provenance import runtime_fingerprints
fresh = runtime_fingerprints(match_config=matchup.match())
print("candidate obs abi", cand.observation_abi_sha256[:12], "fresh", fresh["observation_abi_sha256"][:12])
print("candidate act abi", cand.action_abi_sha256[:12], "fresh", fresh["action_abi_sha256"][:12])
print("artifact_id", cand.artifact_id, "role", cand.role)
key = EvaluationKey(world=managym.WORLD_VERSION, content_suite=matchup.content_suite,
    viewer_boundary="acting-viewer", arena_version="qa310", rating_model_version="unrated",
    rating_prior_sha256=canonical_sha256({}), anchor_cohort_sha256=canonical_sha256(opp.model_dump(mode="json")),
    evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass")
req = EvaluationRequest(key=key, candidate=cand, opponent=opp, checkpoint_paths={"candidate": str(cpath)},
    deal_seeds=(pair.deal,), authored_matchup=matchup, game_seconds=protocol.game_seconds,
    max_commands=protocol.max_commands, purpose="qa-local")
execution = EvalExecution(device="cpu", deadline_unix=time.time() + 900) if mode == "served" else None
t = time.time()
res = evaluate(req, out_dir=out, execution=execution)
print("seconds", round(time.time() - t, 1), "error", res.error)
for r in res.rows:
    print("leg", r.leg, "valid", r.valid, "score_a", r.score_a, "failure", r.failure, "secs", r.game_seconds)
