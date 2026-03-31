# Slow Lamp sensor placement check
# Run this before calibration to find the right sensor distance.
# Checks both ends of the range: full white (saturation risk)
# and dimmest step (noise floor risk).

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
    spi.writebytes(data)
    spi.close()


def fill_all(r, g, b):
    colors = [(int(r), int(g), int(b))] * NUM_LEDS
    send_hd108_colors(colors)


i2c = board.I2C()
sensor = AS7341(i2c)
sensor.gain = 1
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
    return np.mean(readings, axis=0)


if __name__ == "__main__":
    try:
        while True:
            # dark
            print("\n--- Checking sensor placement ---\n")
            fill_all(0, 0, 0)
            sleep(1)
            dark = read_sensor()
            print(f"Dark baseline:  max={int(max(dark))}  {dark.astype(int)}")

            # dimmest step
            fill_all(DIMMEST_STEP, DIMMEST_STEP, DIMMEST_STEP)
            sleep(1)
            dim = read_sensor()
            dim_above_dark = dim - dark
            print(f"Dimmest (={DIMMEST_STEP}): max={int(max(dim))}  above dark={dim_above_dark.astype(int)}")

            # full white
            fill_all(65535, 65535, 65535)
            sleep(1)
            bright = read_sensor()
            peak = int(max(bright))
            print(f"Full white:     max={peak}  {bright.astype(int)}")

            # verdict
            print()
            saturated = peak >= 65535
            too_dim = max(dim_above_dark) < 50

            if saturated and too_dim:
                print("PROBLEM: saturated on bright AND too dim on low end.")
                print("Try reducing sensor.gain instead of moving the sensor.")
            elif saturated:
                print("TOO CLOSE: sensor saturating on full white.")
                print("Move sensor further away.")
            elif too_dim:
                print("TOO FAR: dimmest step is in the noise floor.")
                print("Move sensor closer.")
            else:
                headroom = 100 * (1 - peak / 65535)
                snr = max(dim_above_dark) / max(max(dark), 1)
                print(f"GOOD. Headroom: {headroom:.0f}%  Low-end SNR: {snr:.1f}x")
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
