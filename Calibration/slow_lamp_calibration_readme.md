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
