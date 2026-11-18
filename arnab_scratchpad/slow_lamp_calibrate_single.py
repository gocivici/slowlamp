# Slow Lamp colour calibration, single brightness register.
# Based on Calibration/slow_lamp_calibrate.py. Same method and same JSON
# format, so the lamp loads the output without changes. Differences:
#   - BRIGHTNESS set once at the top
#   - denser sweep, log-spaced from the bottom (41 steps per channel)
#   - stops if the sensor clips, instead of building a table from bad data
#   - writes the JSON to the repo root, where the lamp reads it
#   - honest time estimate
#
# Before running: turn the lamp on (white is fine) and let it run for
# 10-15 minutes so the LEDs warm up. Cold LEDs give out more light than
# warm ones, red especially, so a cold start bends the curves.

from pathlib import Path
from time import sleep
import json
import numpy as np
import spidev
import board
from adafruit_as7341 import AS7341
from scipy.interpolate import PchipInterpolator


# ===========================================================
# Config
# ===========================================================

BRIGHTNESS = 6              # HD108 5-bit register (0-31). Must match the lamp.

NUM_LEDS = 435              # LEDs on the board
SETTLE_TIME = 1.0           # seconds after LED change
NUM_SAMPLES = 5             # sensor reads to average
MAX_VAL = 65535

# 0, then 40 steps evenly spaced on a log scale from 10 to full.
# Gives ~4 steps per doubling, so the dim end is measured as finely as the top.
NUM_LOG_STEPS = 40
SWEEP_STEPS = np.unique(np.concatenate([
    [0],
    np.geomspace(10, MAX_VAL, NUM_LOG_STEPS).round().astype(int),
]))

# mixes to test additivity
ADDITIVITY_MIXES = [
    ("white",     (MAX_VAL, MAX_VAL, MAX_VAL)),
    ("yellow",    (MAX_VAL, MAX_VAL, 0)),
    ("cyan",      (0, MAX_VAL, MAX_VAL)),
    ("magenta",   (MAX_VAL, 0, MAX_VAL)),
    ("25% white", (MAX_VAL // 4, MAX_VAL // 4, MAX_VAL // 4)),
    ("warm",      (MAX_VAL, MAX_VAL // 3, MAX_VAL // 10)),
    ("cool",      (MAX_VAL // 4, MAX_VAL // 3, MAX_VAL)),
]

# the lamp reads slow_lamp_calibration.json from the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "slow_lamp_calibration.json"


class SensorSaturated(Exception):
    pass


# ===========================================================
# HD108 driver
# ===========================================================

def send_hd108_colors(colors_16bit, global_brightness=BRIGHTNESS):
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 2000000

    data = []
    data.extend([0x00] * 8)

    for r16, g16, b16 in colors_16bit:
        if global_brightness <= 0:
            brightness = 0
        else:
            brightness = ((1 << 15)
                          | (global_brightness << 10)
                          | (global_brightness << 5)
                          | global_brightness)

        data.extend([
            (brightness >> 8) & 0xFF, brightness & 0xFF,
            (r16 >> 8) & 0xFF, r16 & 0xFF,
            (g16 >> 8) & 0xFF, g16 & 0xFF,
            (b16 >> 8) & 0xFF, b16 & 0xFF,
        ])

    num_end = 2 * (len(colors_16bit) + 1)
    data.extend([0xFF] * num_end)
    spi.writebytes2(data)
    spi.close()


def fill_all(r, g, b):
    """Set all LEDs to the same colour."""
    colors = [(int(r), int(g), int(b))] * NUM_LEDS
    send_hd108_colors(colors)
    send_hd108_colors(colors)


def leds_off():
    fill_all(0, 0, 0)


# ===========================================================
# AS7341 sensor
# ===========================================================

i2c = board.I2C()
sensor = AS7341(i2c)
sensor.gain = 5          # 16x
sensor.atime = 100       # with default astep 999: ~281 ms per measurement


def read_sensor(n_samples=NUM_SAMPLES):
    """Read 8 spectral channels, averaged. Stops the run if any channel clips."""
    readings = []
    for _ in range(n_samples):
        sample = [
            sensor.channel_415nm, sensor.channel_445nm,
            sensor.channel_480nm, sensor.channel_515nm,
            sensor.channel_555nm, sensor.channel_590nm,
            sensor.channel_630nm, sensor.channel_680nm,
        ]
        if max(sample) >= 65535:
            raise SensorSaturated(
                f"sensor clipped: {sample}. Lower sensor.gain or move the "
                f"sensor further away, then run again.")
        readings.append(sample)
        sleep(0.05)
    return np.mean(readings, axis=0)


def drive_and_read(r, g, b):
    """Fill all LEDs, wait for settle, read sensor."""
    fill_all(r, g, b)
    sleep(SETTLE_TIME)
    return read_sensor()


# ===========================================================
# Sweep one channel
# ===========================================================

def sweep_channel(channel_index, steps=SWEEP_STEPS):
    name = ["Red", "Green", "Blue"][channel_index]
    print(f"\n  Sweeping {name} ({len(steps)} steps)...")
    results = []

    for val in steps:
        rgb = [0, 0, 0]
        rgb[channel_index] = int(val)
        reading = drive_and_read(*rgb)
        total = float(np.sum(reading))
        print(f"    {val:>5d} -> total={total:.0f}  {reading.astype(int)}")
        results.append({
            "drive": int(val),
            "channels": reading.tolist(),
            "total": total,
        })

    return results


# ===========================================================
# Test additivity
# ===========================================================

def test_additivity(dark, r_sweep, g_sweep, b_sweep):
    print("\n  Testing additivity with colour mixes...")

    def make_predictor(sweep_data):
        drives = [s["drive"] for s in sweep_data]
        channels = np.array([s["channels"] for s in sweep_data])
        return [
            PchipInterpolator(drives, channels[:, ch])
            for ch in range(8)
        ]

    r_pred = make_predictor(r_sweep)
    g_pred = make_predictor(g_sweep)
    b_pred = make_predictor(b_sweep)

    mix_results = []
    for name, (rv, gv, bv) in ADDITIVITY_MIXES:
        actual = drive_and_read(rv, gv, bv)

        # predicted = dark + sum of individual contributions
        predicted = np.array(dark)
        for ch in range(8):
            if rv > 0:
                predicted[ch] += r_pred[ch](rv) - dark[ch]
            if gv > 0:
                predicted[ch] += g_pred[ch](gv) - dark[ch]
            if bv > 0:
                predicted[ch] += b_pred[ch](bv) - dark[ch]

        error = actual - predicted
        rel_error = np.linalg.norm(error) / (np.linalg.norm(actual) + 1e-6)

        status = "OK" if rel_error < 0.1 else "NONLINEAR"
        print(f"    {name:>12s}: error={rel_error:.3f}  ({status})")

        mix_results.append({
            "name": name,
            "rgb": (int(rv), int(gv), int(bv)),
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
            "relative_error": round(float(rel_error), 4),
        })

    return mix_results


# ===========================================================
# Build correction LUTs
# ===========================================================

def build_lut(sweep_data, dark_total):
    """
    From a single-channel sweep, build inverse LUTs:
    desired output level (index) -> corrected drive value.

    Returns both a full 65536-entry LUT (direct lookup)
    and a compact 256-entry LUT (interpolate at runtime).
    """
    drives = np.array([s["drive"] for s in sweep_data])
    totals = np.array([s["total"] for s in sweep_data])

    # subtract dark, normalise to 0-1
    totals = np.maximum(totals - dark_total, 0)
    max_output = totals[-1] if totals[-1] > 0 else 1.0
    totals_norm = totals / max_output

    # forward curve: drive -> normalised output
    forward = PchipInterpolator(drives, totals_norm)

    # sample densely and invert
    dense_drives = np.linspace(0, MAX_VAL, 4096)
    dense_outputs = np.clip(forward(dense_drives), 0, 1)

    # enforce monotonic for inversion
    for i in range(1, len(dense_outputs)):
        if dense_outputs[i] <= dense_outputs[i - 1]:
            dense_outputs[i] = dense_outputs[i - 1] + 1e-10

    # inverse: desired output -> drive value
    inverse = PchipInterpolator(dense_outputs, dense_drives)

    full_norm = np.linspace(0, 1, 65536)
    full_lut = np.clip(inverse(full_norm), 0, MAX_VAL).astype(int).tolist()

    compact_norm = np.linspace(0, 1, 256)
    compact_lut = np.clip(inverse(compact_norm), 0, MAX_VAL).astype(int).tolist()

    return full_lut, compact_lut


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    total_measurements = len(SWEEP_STEPS) * 3 + len(ADDITIVITY_MIXES) + 1
    # each sample is two ~281 ms sensor measurements (two banks of 4 channels)
    per_reading = SETTLE_TIME + NUM_SAMPLES * (2 * 0.29 + 0.05)
    est_time = total_measurements * per_reading

    print("=" * 50)
    print("Slow Lamp calibration (single brightness)")
    print(f"LEDs: {NUM_LEDS}")
    print(f"Brightness register: {BRIGHTNESS}/31")
    print(f"Sweep steps per channel: {len(SWEEP_STEPS)}")
    print(f"Total measurements: {total_measurements}")
    print(f"Estimated time: ~{est_time / 60:.0f} min")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)

    try:
        # dark baseline
        print("\n  Reading dark baseline...")
        leds_off()
        sleep(SETTLE_TIME)
        dark = read_sensor()
        dark_total = float(np.sum(dark))
        print(f"  Dark: {dark.astype(int)}  total={dark_total:.0f}")

        # sweep each channel
        r_sweep = sweep_channel(0)
        g_sweep = sweep_channel(1)
        b_sweep = sweep_channel(2)

        # additivity check
        mix_results = test_additivity(dark, r_sweep, g_sweep, b_sweep)

        # build LUTs
        print("\n  Building correction LUTs...")
        r_full, r_compact = build_lut(r_sweep, dark_total)
        g_full, g_compact = build_lut(g_sweep, dark_total)
        b_full, b_compact = build_lut(b_sweep, dark_total)
        print("  Done.")

        max_mix_error = max(m["relative_error"] for m in mix_results)
        additive = max_mix_error < 0.1

        # save
        output = {
            "brightness_register": BRIGHTNESS,
            "num_leds": NUM_LEDS,
            "sweep_steps": SWEEP_STEPS.tolist(),
            "dark": dark.tolist(),
            "sweeps": {
                "red": r_sweep,
                "green": g_sweep,
                "blue": b_sweep,
            },
            "additivity_tests": mix_results,
            "system_is_additive": additive,
            "max_mix_error": round(max_mix_error, 4),
            "luts_full": {
                "red": r_full,
                "green": g_full,
                "blue": b_full,
                "size": 65536,
                "note": "index = desired output (0-65535), value = corrected drive value",
            },
            "luts_compact": {
                "red": r_compact,
                "green": g_compact,
                "blue": b_compact,
                "size": 256,
                "note": "256 points, linearly interpolate at runtime",
            },
        }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f)
        print(f"\n  Saved to {OUTPUT_FILE}")
        print(f"  Additive: {additive} (max mix error: {max_mix_error:.3f})")

        if not additive:
            print("\n  WARNING: system is not cleanly additive.")
            print("  Per-channel LUTs will help but won't fully")
            print("  correct colour mixes. A 3x3 matrix correction")
            print("  on top of the LUTs may be needed.")

        print("\n  Calibration complete.")

    except SensorSaturated as e:
        print(f"\n  STOPPED: {e}")
        print("  Nothing was saved.")
    except KeyboardInterrupt:
        print("\n  Interrupted. Nothing was saved.")
    finally:
        leds_off()
