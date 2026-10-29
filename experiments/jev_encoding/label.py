"""Replay MTG-212 campaign-3 games and label every Jev decision with the checkpoint's view.

For each Jev decision with two or more legal actions, record the checkpoint's
action distribution (by text reference) and value estimate from Jev's seat,
plus Jev's logged reply probabilities. Replay must reproduce each logged capture
digest exactly; any divergence aborts that game.
"""

import argparse
import json
from pathlib import Path

from managym.decision import DecisionFrame, apply_semantic_command
import torch

from manabot.env.match import Match
from manabot.infra.config import MatchConfig
from manabot.model.checkpoint import load_checkpoint
from manabot.semantic.text_observation import capture_text_decision
import managym


def _checkpoint_view(agent, space, env: managym.Env, references: dict[int, str]) -> tuple[dict[str, float], float]:
    raw = env.observation_for_player(env.current_agent_index())
    encoded = space.encode(raw)
    tensors = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in encoded.items()}
    with torch.inference_mode():
        logits, value = agent.forward(tensors)
    probs = torch.softmax(logits[0], dim=-1)
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    out: dict[str, float] = {}
    for index, offer in enumerate(frame.offers):
        if encoded["actions_valid"][index]:
            out[references[int(offer["id"])]] = round(float(probs[index]), 5)
    return out, float(value.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent, space = load_checkpoint(str(args.checkpoint))
    agent.eval()
    rows = 0
    with args.out.open("w") as sink:
        for game in sorted(args.campaign.glob("game-*")):
            if not game.is_dir() or not (game / "summary.json").exists():
                continue
            summary = json.loads((game / "summary.json").read_text())
            if not summary.get("completed") or not summary.get("replay_matches"):
                continue
            config = json.loads((game / "config.json").read_text())
            players = Match(MatchConfig.model_validate(config["match"])).to_rust()
            env = managym.Env(seed=config["seed"], skip_trivial=config["skip_trivial"])
            env.reset(players)
            for step in sorted(game.glob("decision-*")):
                choice = json.loads((step / "choice.json").read_text())
                decision = capture_text_decision(env, history_capacity=config["history_capacity"])
                logged = json.loads((step / "capture.json").read_text())
                if json.loads(decision.capture_json).get("digest") != logged.get("digest"):
                    raise RuntimeError(f"{step}: replay capture diverged")
                if choice["producer"] == "initial" and len(decision.actions) >= 2:
                    references = {a.command.offer_id: a.reference for a in decision.actions}
                    probs, value = _checkpoint_view(agent, space, env, references)
                    selection = json.loads((step / "selection.json").read_text())
                    answer = selection["attempts"][-1]["decision_reply"]["answers"]["action"]
                    sink.write(json.dumps({
                        "game": game.name,
                        "step": step.name,
                        "seat": config["model_seat"],
                        "winner": summary["winner"],
                        "n": len(decision.actions),
                        "kinds": {a.reference: a.description for a in decision.actions},
                        "checkpoint": probs,
                        "checkpoint_value": value,
                        "jev_logged": answer.get("probabilities"),
                        "jev_choice": choice["reference"],
                        "prompt_tokens": selection["attempts"][-1].get("prompt_tokens"),
                    }) + "\n")
                    rows += 1
                apply_semantic_command(env, json.loads((step / "command.json").read_text()))
    print("rows", rows)


if __name__ == "__main__":
    main()
