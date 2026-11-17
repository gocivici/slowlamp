"""
Write a copy of a cover archive.png with the badly encoded records removed.

Bad records = a contiguous run (>= MIN_RUN) of records whose 5 colours all have
blue pinned at 255 (the purple/blue chunk in archive20261009.png, Jun 1-10 2026).
Isolated blue-pinned records are kept; they're real sky colours.

    python3 tools/clean_archive.py Samples/archive20261009.png
    -> Samples/archive20261009_clean.png
"""

import io
import math
import os
import sys
from contextlib import redirect_stdout

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizer", "pipeline_snapshot"))
with redirect_stdout(io.StringIO()):
    import cover

BLOCK_W, BLOCK_H, BLOCKS_PER_ROW = 4, 3, 24  # same as cover.save
COLOR_PIXELS = [1, 4, 5, 6, 9]  # ac1, ac4, vibrant, ac2, ac3
MIN_RUN = 24


def blue_pinned(block):
    return all(block[i][2] == 255 for i in COLOR_PIXELS)


def bad_runs(blocks):
    runs, i = [], 0
    while i < len(blocks):
        if blue_pinned(blocks[i]):
            j = i
            while j + 1 < len(blocks) and blue_pinned(blocks[j + 1]):
                j += 1
            if j - i + 1 >= MIN_RUN:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def write_archive(blocks, filename):
    rows = math.ceil(len(blocks) / BLOCKS_PER_ROW)
    cols = min(len(blocks), BLOCKS_PER_ROW)
    img = Image.new("RGBA", (cols * BLOCK_W, rows * BLOCK_H), (0, 0, 0, 0))
    for i, pixels in enumerate(blocks):
        tile = Image.new("RGBA", (BLOCK_W, BLOCK_H))
        tile.putdata(pixels)
        img.paste(tile, ((i % BLOCKS_PER_ROW) * BLOCK_W, (i // BLOCKS_PER_ROW) * BLOCK_H))
    img.save(filename)


def main(src):
    blocks = cover.get_existing_blocks(src, BLOCK_W, BLOCK_H)
    runs = bad_runs(blocks)
    drop = {k for a, b in runs for k in range(a, b + 1)}
    kept = [b for k, b in enumerate(blocks) if k not in drop]

    root, ext = os.path.splitext(src)
    dst = f"{root}_clean{ext}"
    write_archive(kept, dst)

    for a, b in runs:
        print(f"removed records {a}-{b} ({b - a + 1})")
    print(f"{len(blocks)} -> {len(kept)} records, written to {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
