"""Throwaway baseline: length-delimited JSON and all Python scalar buffers."""
import hashlib
import random
import struct

import managym
from manabot.env.match import Match
from manabot.env.observation import ObservationConfig, ObservationEncoder


def main() -> None:
    digest = hashlib.sha256()
    encoder = ObservationEncoder(ObservationConfig())
    decisions = 0
    def add(value: bytes) -> None:
        digest.update(struct.pack('<Q', len(value)))
        digest.update(value)
    for seed in range(20):
        env = managym.Env(seed=seed)
        obs, _ = env.reset(Match().to_rust())
        rng = random.Random(seed)
        for _ in range(20000):
            add(obs.toJSON().encode())
            for key, buffer in sorted(encoder.encode(obs).items()):
                add(key.encode())
                add(str(buffer.shape).encode())
                add(buffer.dtype.str.encode())
                add(buffer.tobytes())
            decisions += 1
            if obs.game_over:
                break
            obs, _, terminated, truncated, _ = env.step(rng.randrange(len(obs.action_space.actions)))
        else:
            raise RuntimeError(f'seed {seed} exceeded decision bound')
    print(decisions, digest.hexdigest(), flush=True)




def show_tokens() -> None:
    from manabot.semantic.text_observation import capture_text_decision, submit_response
    from manabot.verify.text_observation import priority_root

    env = priority_root((), ("Swamp", "Joo Dee, One of Many"))
    decision = capture_text_decision(env)
    activation = next(a for a in decision.actions if "Activate" in a.description and "Joo Dee" in a.description)
    submit_response(env, decision, activation.reference)
    for _ in range(8):
        decision = capture_text_decision(env)
        if "Stage: surveil." in decision.text:
            break
        submit_response(env, decision, decision.actions[0].reference)
    else:
        raise AssertionError("surveil decision not reached")
    keep = next(a for a in decision.actions if "Keep on top" in a.description)
    submit_response(env, decision, keep.reference)
    decision = capture_text_decision(env)
    source = decision.text.split("Source: ", 1)[1].split(" ", 1)[0]
    original = next(a for a in decision.actions if f"Select {source} " in a.description and "Sacrifice" in a.description)
    submit_response(env, decision, original.reference)
    actor = env.current_agent_index()
    assert actor is not None
    obs = env.observation_for_player(actor)
    tokens = [row for row in obs.agent_objects + obs.opponent_objects if row.is_token]
    assert tokens
    print(type(tokens[0]).__name__, [row.name for row in tokens])
    print(len(obs.agent_cards), managym.Card is managym.Object)


if __name__ == '__main__':
    import sys
    if '--show-tokens' in sys.argv:
        show_tokens()
    else:
        main()
