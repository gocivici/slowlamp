# Slow Lamp Colour Calibration

The HD108 LEDs don't give out light in a straight line with the value you send,
and red, green and blue don't add up to a neutral white. Calibration measures
the panel with an AS7341 sensor and builds a lookup table per colour that the
lamp uses from then on.

## Steps

Run everything from the repo root.

1. **Stop the lamp:** `sudo systemctl stop slowlamp_app`
2. **Place the sensor.** Put it in the dark box facing the diffuser and run
   `python Calibration/slow_lamp_precheck.py`. Move it until it says GOOD,
   then tape it down.
3. **Warm up.** Start the lamp (`sudo systemctl start slowlamp_app`) and leave
   it for 10-15 minutes. Cold LEDs are brighter than warm ones.
4. **Calibrate.** Stop the lamp and straight away run
   `python Calibration/slow_lamp_calibrate_single.py` (about 9 minutes).
5. **Match white by eye.** Run `python Calibration/slow_lamp_white_nudge.py`
   and nudge the panel until it matches a MacBook showing a white page.
   Setup and keys are in the script's header.
6. **Set the lamp's white.** Run `python Calibration/slow_lamp_white_point.py`.
   It shifts your match to 4200 K, the camera's white balance, so the panel
   will look warmer than the MacBook afterwards.
7. **Start the lamp again.**

To redo steps 5-6, copy `slow_lamp_calibration_before_white_point.json` back
over `slow_lamp_calibration.json` first.

## Keep these the same

- Brightness register: whatever the lamp runs at. It's set at the top of
  the precheck (`GLOBAL_BRIGHTNESS`), the calibration and the nudge script
  (`BRIGHTNESS`), and all three have to match the lamp. The tables only hold
  at the brightness they were measured at, so change it and you need to
  recalibrate.
- Sensor position, gain and integration time during a run.
- No light leaking into the box.

## How it works

- **Tables (step 4).** The script measures each colour at 41 drive values,
  fits a curve and inverts it, so "half the light" really gives half the
  light. Each table is scaled to its own colour's maximum, so it doesn't
  change the balance between red, green and blue.
- **White (steps 5-6).** The nudge script sets that balance by eye against
  the MacBook (D65). The white point script then uses the sensor readings
  only for the step from D65 to 4200 K. The sensor's colour conversion comes
  from datasheet values, so it's only trusted for that small step.
- **At runtime** the lamp sends each colour's linear light level (0-65535)
  through its table. It has to be linear, not gamma-encoded sRGB.
- **Mix test.** The calibration also checks a few colour mixes against the
  sum of the single colours. If they report NONLINEAR, the colours affect
  each other (heat, power supply sag) and the tables alone won't make mixes
  exact.

## Files

| File | What it does |
|---|---|
| `slow_lamp_precheck.py` | Checks the sensor distance (step 2) |
| `slow_lamp_calibrate_single.py` | Builds the tables at one brightness (step 4) |
| `slow_lamp_white_nudge.py` | Matches white to a MacBook by eye (step 5) |
| `slow_lamp_white_point.py` | Sets white to 4200 K (step 6) |
| `slow_lamp_calibrate.py` | Older version: fewer steps, saves to whatever folder it's run from |
| `slow_lamp_apply_lut.py` | Example of using the tables at runtime |
| `test_lut_with_window.py` | Compares camera colours with and without the tables |
| `AS7341_DS000504_v3-00.pdf` | Sensor datasheet |

Output: `slow_lamp_calibration.json` in the repo root, where the lamp reads it.
