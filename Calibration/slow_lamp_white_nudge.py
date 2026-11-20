# Slow Lamp white nudge
# Run after the colour calibration, before slow_lamp_white_point.py.
# Holds white on the panel so you can match it by eye to a MacBook showing
# a white page, and nudge it warmer/cooler and greener/pinker until it does.
# Your match is saved into slow_lamp_calibration.json as "eye_gains", and
# the white point script then shifts it from the MacBook's white to the
# lamp's 4200 K.
#
# MacBook pages (paste into the address bar, full screen):
#   white: data:text/html,<body style="background:%23ffffff">
#   grey:  data:text/html,<body style="background:%23808080">
#
# Keys:
#   a / d   warmer / cooler        (A / D for a bigger step)
#   w / s   greener / pinker       (W / S for a bigger step)
#   space   switch between your match and the panel's own white
#   g       switch between white and mid grey
#   r       reset
#   Enter   save and quit
#   q       quit without saving
#
# Usage (from the repo root, with the lamp's service stopped):
#   python Calibration/slow_lamp_white_nudge.py

import json
import sys
import numpy as np


CAL_FILE = sys.argv[1] if len(sys.argv) > 1 else "slow_lamp_calibration.json"

BRIGHTNESS = 6      # HD108 register (0-31). Must match the lamp and the calibration.
NUM_LEDS = 435

SMALL_STEP = 0.01   # about 1% per press
BIG_STEP = 0.05

GREY = 0.216        # sRGB 128 as linear light, matches the grey MacBook page


def gains_from(warm, green):
    """Turn the two nudges into red, green, blue gains (largest = 1)."""
    g = np.exp(np.array([
        warm - green / 2,     # red
        green,                # green
        -warm - green / 2,    # blue
    ]))
    return g / g.max()


def send(spi, rgb16):
    """Fill all LEDs with one 16-bit colour (HD108 protocol)."""
    b = (1 << 15) | (BRIGHTNESS << 10) | (BRIGHTNESS << 5) | BRIGHTNESS
    r, g, bl = (int(v) for v in rgb16)
    led = [b >> 8, b & 0xFF, r >> 8, r & 0xFF, g >> 8, g & 0xFF, bl >> 8, bl & 0xFF]
    data = [0x00] * 8 + led * NUM_LEDS + [0xFF] * (2 * (NUM_LEDS + 1))
    spi.writebytes2(data)
    spi.writebytes2(data)


def drive(luts, level, gains):
    """What the lamp would send for this level of white, through the LUTs."""
    return [luts[c][int(round(level * g * 65535))] for c, g in zip(("red", "green", "blue"), gains)]


def read_key():
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    import spidev

    with open(CAL_FILE) as f:
        cal = json.load(f)
    if "white_point_gains" in cal:
        sys.exit(f"{CAL_FILE} already has white point gains. Copy "
                 f"slow_lamp_calibration_before_white_point.json back over it first.")

    luts = cal["luts_full"]
    saved = cal.get("eye_gains", {})
    warm, green = saved.get("warm", 0.0), saved.get("green", 0.0)
    level, show_native = 1.0, False

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 2000000

    print("a/d warmer/cooler, w/s greener/pinker (capitals for bigger steps), "
          "space before/after, g white/grey, r reset, Enter save, q quit\n")
    try:
        while True:
            gains = np.ones(3) if show_native else gains_from(warm, green)
            send(spi, drive(luts, level, gains))
            print(f"\r{'panel own white' if show_native else 'your match     '}  "
                  f"{'white' if level == 1.0 else 'grey '}  "
                  f"warm {warm:+.2f}  green {green:+.2f}  "
                  f"gains R {gains[0]:.3f} G {gains[1]:.3f} B {gains[2]:.3f}   ",
                  end="", flush=True)

            k = read_key()
            step = BIG_STEP if k.isupper() else SMALL_STEP
            k = k.lower()
            if k == "a":
                warm += step
            elif k == "d":
                warm -= step
            elif k == "w":
                green += step
            elif k == "s":
                green -= step
            elif k == " ":
                show_native = not show_native
            elif k == "g":
                level = GREY if level == 1.0 else 1.0
            elif k == "r":
                warm, green = 0.0, 0.0
            elif k in ("\n", "\r"):
                g = gains_from(warm, green)
                cal["eye_gains"] = {"gains": g.tolist(), "warm": warm, "green": green}
                with open(CAL_FILE, "w") as f:
                    json.dump(cal, f)
                print(f"\n\nSaved to {CAL_FILE}. Now run slow_lamp_white_point.py with TARGET = \"lamp\".")
                break
            elif k == "q":
                print("\n\nQuit without saving.")
                break
    finally:
        send(spi, (0, 0, 0))
        spi.close()
