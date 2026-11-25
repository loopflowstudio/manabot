"""Throwaway Academy with the Reader fixture policy, for the live-analysis browser proof."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ACADEMY_FRONTEND_BUILD", str(Path(__file__).resolve().parents[2] / "academy/ui/build"))
from managym.academy.launch import serve
from tests.academy.test_played_analysis import ReadingFactory

academy = serve(port=int(sys.argv[1]), traces_dir=Path(sys.argv[2]), policies={"Reader": ReadingFactory()})
print(academy.url, flush=True)
try:
    while True:
        time.sleep(1)
finally:
    academy.close()
