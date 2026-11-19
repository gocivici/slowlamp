# Slow Lamp white point
# Run after the colour calibration. Reads slow_lamp_calibration.json, works
# out what white the panel makes at full red + green + blue, and scales the
# three LUTs so full white lands on D65 (the white sRGB assumes).
#
# No LEDs or sensor needed: it uses the AS7341 readings for full red, full
# green and full blue that the calibration already saved.
#
# Usage (from the repo root, where the lamp reads the file):
#   python Calibration/slow_lamp_white_point.py
# A copy of the original file is kept as slow_lamp_calibration_before_white_point.json.
#
# How it works:
#   1. Model each AS7341 band as a Gaussian, using the datasheet's centre,
#      width and relative responsivity.
#   2. Fit a 3x8 matrix that turns the 8 band readings into CIE XYZ.
#   3. Convert full red, green and blue to XYZ and solve for the three gains
#      (largest = 1) that make their sum D65.
#   4. Scale each LUT by its gain. The LUT index is a linear light level, so
#      gain g on a channel means lut[i] -> lut[i * g].
#
# Accuracy: the AS7341 is a spectral sensor, not a colorimeter, so step 2 is an
# approximation. The band values below are typical, theoretical values from
# the datasheet, not measured on our sensor, so they might be wrong for it.
# They were checked against the vector data in Figure 19 (all within 0.01).
# But the datasheet's own count tables (Figures 8-15, 2700K LED) don't fully
# agree with Figure 19: on a simulated panel the result lands near D65 if the
# sensor behaves like Figure 19, and noticeably warm (x~0.35, y~0.35, towards
# red/yellow) if it behaves like the count tables. Always check the result by
# eye against a D65 monitor (e.g. a MacBook with True Tone and Night Shift
# off) and restore the backup if it looks worse.
#
# References:
#   - ams OSRAM AS7341 datasheet DS000504 v3-00 (2020-06-25),
#     Calibration/AS7341_DS000504_v3-00.pdf:
#     Figure 7 (centre wavelength, FWHM) and Figure 19 (responsivity relative
#     to F8). Figure 19 is a graph; peak values below are read off it by eye.
#   - CIE 1931 colour matching functions, analytic approximation from
#     Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ
#     Color Matching Functions", JCGT 2(2), 2013.
#   - D65 white point x=0.3127, y=0.3290 (IEC 61966-2-1, sRGB).

import json
import shutil
import sys
import numpy as np


CAL_FILE = sys.argv[1] if len(sys.argv) > 1 else "slow_lamp_calibration.json"
BACKUP_FILE = CAL_FILE.replace(".json", "_before_white_point.json")

D65_XY = (0.3127, 0.3290)

# AS7341 F1-F8: centre [nm], FWHM [nm] (datasheet Fig. 7),
# peak responsivity relative to F8 (datasheet Fig. 19, read by eye)
BANDS = [
    (415, 26, 0.16),
    (445, 30, 0.30),
    (480, 36, 0.37),
    (515, 39, 0.48),
    (555, 39, 0.58),
    (590, 40, 0.65),
    (630, 50, 0.75),
    (680, 52, 1.00),
]

WL = np.arange(380, 781, 1.0)   # visible range, 1 nm steps


def band_responses():
    """8 x N matrix: modelled responsivity of each band over WL."""
    rows = []
    for centre, fwhm, peak in BANDS:
        sigma = fwhm / 2.3548   # FWHM = 2*sqrt(2*ln 2)*sigma
        rows.append(peak * np.exp(-0.5 * ((WL - centre) / sigma) ** 2))
    return np.array(rows)


def cie_1931():
    """3 x N matrix: CIE 1931 x-bar, y-bar, z-bar over WL (Wyman et al. 2013)."""
    def g(mu, s1, s2):
        s = np.where(WL < mu, s1, s2)
        return np.exp(-0.5 * ((WL - mu) / s) ** 2)

    x = 1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2)
    y = 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)
    z = 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)
    return np.array([x, y, z])


# least-squares fit: XYZ = M @ readings
M = cie_1931() @ np.linalg.pinv(band_responses())


def xy(XYZ):
    return XYZ[0] / XYZ.sum(), XYZ[1] / XYZ.sum()


def xy_to_XYZ(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


if __name__ == "__main__":
    with open(CAL_FILE) as f:
        cal = json.load(f)

    if "white_point_gains" in cal:
        sys.exit(f"{CAL_FILE} already has white point gains. Run on a fresh calibration.")

    dark = np.array(cal["dark"])
    primaries = {}
    for name in ("red", "green", "blue"):
        full = np.array(cal["sweeps"][name][-1]["channels"])   # last step = full drive
        primaries[name] = M @ (full - dark)

    white = sum(primaries.values())
    print(f"Panel white now:  x={xy(white)[0]:.4f}  y={xy(white)[1]:.4f}")
    print(f"Target (D65):     x={D65_XY[0]:.4f}  y={D65_XY[1]:.4f}")

    # solve gains so gr*R + gg*G + gb*B = D65 (up to brightness)
    A = np.column_stack([primaries["red"], primaries["green"], primaries["blue"]])
    gains = np.linalg.solve(A, xy_to_XYZ(*D65_XY))
    if np.any(gains <= 0):
        sys.exit(f"Got non-positive gains {gains}. The sensor-to-XYZ conversion is "
                 f"probably off for these LEDs; nothing was changed.")
    gains = gains / gains.max()
    print(f"Gains:            red={gains[0]:.3f}  green={gains[1]:.3f}  blue={gains[2]:.3f}")

    # scale the LUTs: index is linear light, so look up i * gain
    idx = np.arange(65536)
    compact_idx = np.linspace(0, 65535, 256).round().astype(int)
    for name, gain in zip(("red", "green", "blue"), gains):
        old = np.array(cal["luts_full"][name])
        new = old[np.round(idx * gain).astype(int)]
        cal["luts_full"][name] = new.tolist()
        cal["luts_compact"][name] = new[compact_idx].tolist()

    cal["white_point_gains"] = {"red": float(gains[0]), "green": float(gains[1]),
                                "blue": float(gains[2]), "target_xy": D65_XY}

    shutil.copyfile(CAL_FILE, BACKUP_FILE)
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f)
    print(f"\nSaved to {CAL_FILE} (original kept as {BACKUP_FILE})")
