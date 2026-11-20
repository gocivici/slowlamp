# Slow Lamp sensor placement check
# Run this before calibration to find the right sensor distance.
# Checks both ends of what the calibration measures:
#   - full white, the brightest thing it shows (saturation risk)
#   - red, green and blue each alone at the dimmest step, the dimmest
#     things it measures (noise floor risk)

from time import sleep
import numpy as np
import spidev
import board
from adafruit_as7341 import AS7341


# --- must match your calibration config ---
NUM_LEDS = 436
GLOBAL_BRIGHTNESS = 6
DIMMEST_STEP = 10       # lowest non-zero drive value in your sweep


def send_hd108_colors(colors_16bit, global_brightness=GLOBAL_BRIGHTNESS):
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
    colors = [(int(r), int(g), int(b))] * NUM_LEDS
    send_hd108_colors(colors)
    send_hd108_colors(colors)


i2c = board.I2C()
sensor = AS7341(i2c)
sensor.gain = 5
sensor.atime = 100

CHANNEL_NAMES = [
    "415nm", "445nm", "480nm", "515nm",
    "555nm", "590nm", "630nm", "680nm",
]


def read_sensor():
    readings = []
    for _ in range(5):
        readings.append([
            sensor.channel_415nm, sensor.channel_445nm,
            sensor.channel_480nm, sensor.channel_515nm,
            sensor.channel_555nm, sensor.channel_590nm,
            sensor.channel_630nm, sensor.channel_680nm,
        ])
        sleep(0.05)
    readings = np.array(readings)
    # also return the highest single reading: the calibration stops if any
    # one reading clips, so the clipping check has to look at each one
    return readings.mean(axis=0), readings.max()


if __name__ == "__main__":
    try:
        while True:
            # dark
            print("\n--- Checking sensor placement ---\n")
            fill_all(0, 0, 0)
            sleep(3)
            dark, _ = read_sensor()
            print(f"Dark baseline:  max={int(max(dark))}  {dark.astype(int)}")

            # each colour alone at the dimmest step (the sweep measures them one at a time)
            dim_peaks = {}
            for name, rgb in [("red",   (DIMMEST_STEP, 0, 0)),
                              ("green", (0, DIMMEST_STEP, 0)),
                              ("blue",  (0, 0, DIMMEST_STEP))]:
                fill_all(*rgb)
                sleep(3)
                above_dark = read_sensor()[0] - dark
                dim_peaks[name] = max(above_dark)
                print(f"Dim {name:<5} (={DIMMEST_STEP}): above dark max={int(dim_peaks[name])}  {above_dark.astype(int)}")

            # full white
            fill_all(65535, 65535, 65535)
            sleep(3)
            bright, bright_max = read_sensor()
            peak = int(max(bright))
            print(f"Full white:     max={peak}  {bright.astype(int)}")

            # verdict
            print()
            saturated = bright_max >= 65535
            too_dim = [name for name, p in dim_peaks.items() if p < 50]

            if saturated and too_dim:
                print(f"PROBLEM: saturated on bright AND too dim on {', '.join(too_dim)}.")
                print("This shouldn't really happen at our brightness. Don't calibrate; pass this output on.")
            elif saturated:
                print("TOO CLOSE: sensor saturating on full white.")
                print("Move sensor further away.")
            elif too_dim:
                print(f"TOO FAR: {', '.join(too_dim)} at the dimmest step is in the noise floor.")
                print("Move sensor closer.")
            else:
                headroom = 100 * (1 - peak / 65535)
                weakest = min(dim_peaks, key=dim_peaks.get)
                snr = dim_peaks[weakest] / max(max(dark), 1)
                print(f"GOOD. Headroom: {headroom:.0f}%  Low-end SNR: {snr:.1f}x (weakest: {weakest})")
                print("Tape it down and run calibration.")

            fill_all(0, 0, 0)

            resp = input("\nPress Enter to re-check, or 'q' to quit: ").strip()
            if resp.lower() == 'q':
                break

    except KeyboardInterrupt:
        pass
    finally:
        fill_all(0, 0, 0)
        print("Done.")
