"""
Check that workers.interpolate_two_frames_batched gives the same animation plan as the
production interpolate_two_frames, on consecutive records from a sample.

    python3 check_batched.py ../Samples/archive20261009_clean.png [pairs]
"""

import sys
import time
import random

import numpy as np

import engine
import workers
from engine import anim


def main(path, pairs=40):
    _, recs, _ = engine.parse_sample(path, path)
    for layout in engine.CLUSTER_LAYOUTS:
        engine.set_cluster_layout(layout)
        random.seed(0)
        np.random.seed(0)
        pb = engine.Playback.__new__(engine.Playback)
        pb.records = recs[: pairs + 1]
        tracks = []
        anim.init_tracks(pb._trace(pb.records[0]), tracks)
        kfs = [pb._keyframe(k, tracks) for k in range(pairs + 1)]

        worst, t_orig, t_batch = 0.0, 0.0, 0.0
        for k in range(pairs):
            for length in (18, 4, 1):
                t = time.time()
                a = anim.interpolate_two_frames(kfs[k], kfs[k + 1], length)
                t_orig += time.time() - t
                t = time.time()
                b = workers.interpolate_two_frames_batched(kfs[k], kfs[k + 1], length)
                t_batch += time.time() - t
                worst = max(worst, np.abs(a - b).max())
                if not np.array_equal(workers.to_drive(a), workers.to_drive(b)):
                    print(f"{layout}: SPI values differ at pair {k}, length {length}")
        print(f"{layout}: {pairs} pairs x 3 lengths, max |plan diff| = {worst}, "
              f"original {t_orig:.1f}s, batched {t_batch:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
