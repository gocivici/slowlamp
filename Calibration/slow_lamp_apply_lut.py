# Apply calibration LUTs at runtime
# Two options: full (direct lookup) or compact (interpolated)

import json
import numpy as np


with open("slow_lamp_calibration.json") as f:
    cal = json.load(f)


# --- Option A: full LUT, direct lookup, ~1.5MB in memory ---

r_full = cal["luts_full"]["red"]
g_full = cal["luts_full"]["green"]
b_full = cal["luts_full"]["blue"]


def correct_full(r, g, b):
    """Direct lookup, no math at runtime."""
    return r_full[r], g_full[g], b_full[b]


# --- Option B: compact LUT, interpolated, ~5KB in memory ---

r_compact = np.array(cal["luts_compact"]["red"])
g_compact = np.array(cal["luts_compact"]["green"])
b_compact = np.array(cal["luts_compact"]["blue"])


def correct_compact(r, g, b):
    """Interpolate from 256-point LUT."""
    return (
        int(np.interp(r, np.linspace(0, 65535, 256), r_compact)),
        int(np.interp(g, np.linspace(0, 65535, 256), g_compact)),
        int(np.interp(b, np.linspace(0, 65535, 256), b_compact)),
    )


# example
desired = (30000, 10000, 50000)
print(f"Desired:       {desired}")
print(f"Full lookup:   {correct_full(*desired)}")
print(f"Compact interp:{correct_compact(*desired)}")
