# Slow Lamp Colour Calibration

## The problem

When you send RGB values to the HD108 LEDs, the actual light output doesn't match
what you asked for. LEDs are nonlinear (especially at low drive values), and the
three colour channels have different efficiencies. So (30000, 30000, 30000) doesn't
produce neutral white, and the relationship between drive value and brightness isn't
a straight line.

## The fix

Place the lamp and the AS7341 sensor inside a dark box, with the sensor facing the
lamp's diffuser at a fixed distance. Run a calibration routine that measures what
the lamp actually produces at a range of drive values. Because the sensor is reading
through the diffuser, the calibration captures the full optical path: LED emission,
any colour shift from the diffuser material, and the mixing behaviour of light
inside the enclosure. Build a correction lookup table that maps "what you want" to
"what you need to send." Remove the sensor. The lamp uses the correction data from
then on.

## What the calibration does

1. Reads a dark baseline (all LEDs off, sensor noise floor).

2. Sweeps red from 0 to 65535 across 20 steps. The steps are concentrated at the
   low end (0, 10, 50, 150, 400) where nonlinearity is worst, then log-spaced
   through the upper range. At each step, all 436 LEDs are set to the same value,
   the sensor settles for 1s, and 5 readings are averaged.

3. Repeats for green and blue.

4. Flashes 7 colour mixes (white, yellow, cyan, magenta, 25% white, warm, cool)
   and compares the sensor reading against what you'd predict by adding the
   individual R, G, B responses together. If they match, the system is additive
   and per-channel correction is sufficient. If they don't, there's crosstalk or
   nonlinearity in the enclosure that would need a more complex correction.

5. For each channel, fits a smooth curve through the (drive value, sensor response)
   data points, then inverts it. The inverse curve answers: "to get X amount of
   light, send Y as the drive value." This is stored as a lookup table.

6. Saves everything to slow_lamp_calibration.json: the raw sweep data, the
   additivity test results, and two versions of the correction LUTs.

## The two LUT sizes

Full (65536 entries per channel, ~1.5MB total): index directly with the desired
value, get the corrected drive value back. No math at runtime.

Compact (256 entries per channel, ~5KB total): linearly interpolate at runtime.
Negligible accuracy loss since the correction curve is smooth. Better for the Pi
where memory is limited.

## How to use at runtime

Load the JSON once at boot. Before sending any colour to the LEDs, pass each
channel value through the LUT:

    desired_r, desired_g, desired_b = 30000, 10000, 50000
    drive_r = r_lut[desired_r]    # full LUT
    drive_g = g_lut[desired_g]
    drive_b = b_lut[desired_b]
    send_hd108_colors([(drive_r, drive_g, drive_b)] * NUM_LEDS)

That's the only change to your existing lamp code.

## Configuration

These must stay the same between calibration and runtime:

- Global brightness register (set to 6 in the script)
- Number of LEDs (436)

These must stay fixed during the calibration run:

- Sensor position and angle relative to the lamp (don't bump it mid-sweep)
- Sensor gain and integration time (pinned in the script)
- Dark box sealed, no ambient light leaking in

If you change the brightness register, recalibrate.

## What the additivity test tells you

If all mixes report "OK" (error < 10%), per-channel LUTs are all you need.

If some report "NONLINEAR", there's interaction between channels. This could be
thermal (436 LEDs warming up when all three channels are on), optical (light
mixing inside the lamp or colour shift from the diffuser), or electrical (power
supply sag under full white). The LUTs will still improve things significantly,
but for perfect mixes you'd need a 3x3 colour correction matrix on top, which
is a separate step.

## Files

slow_lamp_precheck.py    Run first. Checks sensor distance is right.
slow_lamp_calibrate.py   Run once with sensor taped down. Takes ~90 seconds.
slow_lamp_apply_lut.py   Shows how to load and use the LUTs at runtime.
slow_lamp_calibration.json   Generated output. Ship this with the lamp.
slow_lamp_calibrate_single.py   Newer calibration at one brightness. See below.
slow_lamp_white_point.py        Optional. Sets full white to D65. See below.
AS7341_DS000504_v3-00.pdf       Sensor datasheet, used by the white point script.

## Updated workflow (November 2026)

The lamp runs at a single brightness register, so there's now a calibration
script for just that one. The steps are:

1. Stop the lamp so it isn't driving the LEDs at the same time:
   `sudo systemctl stop slowlamp_app`

2. Run `slow_lamp_precheck.py`. It now checks full white for clipping and each
   colour on its own at the dimmest step for noise, and tells you which colour
   failed.

3. Let the lamp run for 10-15 minutes so the LEDs warm up. Cold LEDs give out
   more light than warm ones, red especially.

4. Run the calibration from the repo root:
   `python Calibration/slow_lamp_calibrate_single.py`
   The brightness register is set at the top (`BRIGHTNESS = 6`) and has to match
   the lamp. It uses 0 plus 40 log-spaced steps per channel, so the dim end is measured
   as finely as the top. It takes about 9 minutes, writes
   `slow_lamp_calibration.json` to the repo root, and stops without saving if
   the sensor clips.

5. Compare the panel's full white with a MacBook showing a white page, in a dark
   room, with True Tone, Night Shift and auto-brightness off. Only if there's a
   clear tint, run from the repo root:
   `python Calibration/slow_lamp_white_point.py`
   It uses the readings the calibration already saved (no LEDs or sensor
   needed), scales the three LUTs so full white lands on D65, and keeps the
   original as `slow_lamp_calibration_before_white_point.json`. Check against
   the MacBook again afterwards and copy the original back if it looks worse.

Worth knowing: the LUTs only straighten each channel's curve. Each one is
normalised to its own maximum, so they don't change the balance between red,
green and blue. That balance is what the white point step sets. The white point
script converts the sensor's 8 bands to colour using a model built from the
datasheet, so it's an approximation, which is why the MacBook check comes first.

The LUT index is linear light, so the lamp has to send linear values into it,
not gamma-encoded sRGB.
