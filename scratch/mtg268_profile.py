"""Retain full separate cProfile evidence from the unchanged fixed-position harness."""

import cProfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bench_local_update as bench

original_profile_costs = bench.profile_costs
profile_index = 0


def retain_profile(profile: cProfile.Profile) -> dict[str, bench.Cost]:
    global profile_index
    label = ('before', 'after')[profile_index]
    profile_index += 1
    stats = profile.getstats()
    profile.dump_stats(f'scratch/mtg-268-baseline-{label}.pstats')
    Path(f'scratch/mtg-268-baseline-{label}-profile.json').write_text(
        json.dumps(
            {
                'side': label,
                'total_self_seconds': sum(entry.inlinetime for entry in stats),
                'top_self': [
                    {
                        'function': entry.code if isinstance(entry.code, str) else (
                            f'{entry.code.co_filename}:{entry.code.co_firstlineno}:'
                            f'{entry.code.co_qualname}'
                        ),
                        'calls': entry.callcount,
                        'seconds': entry.totaltime,
                        'self_seconds': entry.inlinetime,
                    }
                    for entry in sorted(stats, key=lambda item: item.inlinetime, reverse=True)[:40]
                ],
            }, indent=2,
        ) + '\n'
    )
    return original_profile_costs(profile)


bench.profile_costs = retain_profile
bench.main()
