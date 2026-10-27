"""Run one inspectable decision from explicit JSON input and inference settings."""

import argparse
from pathlib import Path

from manabot.llm.runner import run_decision
from manabot.llm.schema import DecisionInput, InferenceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("transformers", "openrouter"), default="transformers"
    )
    args = parser.parse_args()
    decision = DecisionInput.model_validate_json(args.input.read_text())
    if args.backend == "openrouter":
        from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession

        hosted_config = OpenRouterConfig.model_validate_json(args.config.read_text())
    else:
        config = InferenceConfig.model_validate_json(args.config.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the destination before inference; never replace prior evidence.
    with args.out.open("x") as output:
        if args.backend == "openrouter":
            with OpenRouterSession(hosted_config) as session:
                result = session.decide(decision)
        else:
            result = run_decision(decision, config)
        output.write(result.model_dump_json(indent=2) + "\n")
    print(f"{result.origin or result.failure.kind}: {args.out}")
    raise SystemExit(0 if result.failure is None else 1)


if __name__ == "__main__":
    main()
