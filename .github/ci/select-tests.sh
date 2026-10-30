#!/usr/bin/env bash
# Select independent CI jobs from NUL-delimited changed paths. Unknown inputs
# run everything; docs and unrelated packages do not fan out into browser tests.
set -euo pipefail
jobs=(rust python_unit integration academy_integration protocol_v1 semantic_conformance frontend clean_machine_play)
for job in "${jobs[@]}"; do declare "$job=false"; done
select_jobs() { local job; for job in "$@"; do printf -v "$job" true; done; }
all() { select_jobs "${jobs[@]}"; }
seen=false
if [[ "${1:-}" == --all ]]; then
  all
else
  while IFS= read -r -d '' file; do
    seen=true
    case "$file" in
      .github/ci/*|.github/workflows/ci.yml|pyproject.toml|uv.lock|managym/Cargo.toml|managym/Cargo.lock|managym/pyproject.toml)
        all ;;
      .github/workflows/manabot-integration.yml) select_jobs integration ;;
      .github/workflows/academy-integration.yml) select_jobs academy_integration ;;
      .github/workflows/*) ;; # Separate workflows own their own triggers.
      scripts/audit_rules.py|docs/rules/audit-2026-10/*)
        select_jobs python_unit ;;
      content/DECKS.md|AGENTS.md|README.md|VISUAL_DESIGN.md|WORLDS.md|*/README.md) ;;
      content/*|conformance/*|manabot/managym/*|managym/src/*|managym/*.py|managym/*.pyi)
        all ;;
      docs/rules/elemental/decks.json|docs/rules/elemental/cube-snapshot.csv) select_jobs academy_integration ;;
      protocol/*) select_jobs rust protocol_v1 frontend semantic_conformance ;;
      managym/tests/*) select_jobs rust semantic_conformance ;;
      academy/ui/src/lib/packs/*|academy/ui/src/lib/fixtures/*)
        select_jobs rust integration academy_integration protocol_v1 frontend semantic_conformance clean_machine_play ;;
      academy/ui/package.json|academy/ui/package-lock.json|academy/ui/*config.*|academy/ui/scripts/*|academy/ui/static/*|academy/ui/src/lib/card-art/*)
        select_jobs frontend clean_machine_play ;;
      academy/ui/*) select_jobs frontend ;;
      academy/fixtures/*|academy/server.py|academy/presentation.py|academy/play_demo.py|academy/curated_pack.py|academy/villain.py|academy/presets.py)
        select_jobs academy_integration protocol_v1 frontend ;;
      academy/*) select_jobs academy_integration protocol_v1 ;;
      tests/academy/test_play_launcher.py|tests/academy/test_clean_machine_verifier.py)
        select_jobs python_unit clean_machine_play ;;
      tests/academy/*) select_jobs academy_integration protocol_v1 ;;
      scripts/play|scripts/play.py|scripts/verify-clean-machine|deploy/play/*)
        select_jobs python_unit protocol_v1 clean_machine_play ;;
      scripts/play-elemental|scripts/play_elemental.py)
        select_jobs academy_integration clean_machine_play ;;
      manabot/env/*|manabot/model/world.py|manabot/sim/search/flat.py|manabot/sim/search/flat_mc.py|manabot/sim/action/structured.py)
        select_jobs integration academy_integration protocol_v1 frontend ;;
      tests/infra/*) select_jobs python_unit ;;
      manabot/infra/*) select_jobs python_unit integration ;;
      manabot/*|tests/semantic/*|tests/llm/*|tests/agent/*|tests/model/*|tests/env/*|tests/eval/*|tests/sim/*|tests/training/*|tests/bench/*|tests/belief/*|tests/verify/*|tests/test_package_boundaries.py|scripts/*|experiments/*|ops/*|docs/experiment-metrics.md|docs/benchmarks/*)
        select_jobs integration ;;
      website/*) select_jobs python_unit ;;
      paper/*|wave/*|scratch/*|.lf/*|docs/*|*.md|LICENSE|.gitignore|.dockerignore) ;;
      *) all ;;
    esac
  done
  [[ "$seen" == true ]] || all
fi
for job in "${jobs[@]}"; do printf '%s=%s\n' "$job" "${!job}"; done
