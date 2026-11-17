# Lamp Replay

Replays field samples (trace log `.txt` or cover `archive.png`) through the lamp's
production animation code and shows what the 436 HD108 LEDs do.

```
cd arnab_scratchpad/visualizer && python3 server.py      # http://localhost:5002
```

Needs: `flask numpy scipy pandas opencv-python pillow scikit-learn colour-science colormath`.

## Using it

1. **Load** a `.txt` trace log or an `archive*.png`.
2. **Sampling** (optional) picks which records play:
   - **All**: every record.
   - **Filter**: date range, hour of day, every Nth record.
   - **Moon**: the records `moon_helper.find_matched_indices` returns for a given lamp time.
3. **interval (s)**: how long each fade between two records takes. Frames run at 60 fps,
   so 0.3 s = 18 frames.
4. **groups on**: the LED positions the colour groups are computed on. *PCB rings*
   (`get_led_points_pcb`, 435 LEDs) matches the board. *23cm grid* is what
   `fgc_arnab.py` line 159 uses now (436 LEDs); it was switched back from the rings in `30cc068`.
5. **Build playback**, then **Play / Pause** and **Restart**.

The panel always draws LEDs at their PCB ring positions in data-chain order. Hover an LED
for its SPI values, or the strip for a record.

## How it works

Keyframes come from `init_tracks` + `generate_frame_from_trace_tracks` (fresh tracks,
fixed random seed so replays repeat), in order, because the colour tracks carry state.
Each fade is `interpolate_two_frames`: a linear OKLab lerp per LED, then
`oklab_to_srgb_255` with the perceptual gamut mapping. Frames are converted to exactly what
`send_hd108_colors_with_brightness` writes: int truncation, 5-bit brightness, 16-bit RGB.

`workers.interpolate_two_frames_batched` is the same function with the colour conversion
done once for all LEDs instead of once per LED; `check_batched.py` confirms identical
output to the production function on real records. Fades are prepared in a process pool
for a window ahead of the playhead.

All colour math, LED clustering, OKLab interpolation and gamut mapping come from the frozen
copies under `pipeline_snapshot/` (see `SNAPSHOT.md`).

## Data cleaning

- `.txt`: empty captures (white, count 0) are skipped, as in
  `light_simulation/record_to_year_new.py`. They never reach `archive.png`.
- Both formats: a run of 24+ records whose 5 colours all have blue pinned at 255 is
  removed. That's the badly encoded purple chunk in `archive20261009.png`
  (207 records, Jun 1-10 2026). `../tools/clean_archive.py` writes a cleaned copy of a PNG.

## Display

On-screen colour = drive value / (255 × 255) as sRGB. With "Apply global brightness" on,
it is also scaled in linear light by brightness / 6 (the clip ceiling). "Diffuser glow"
blurs each LED over ~14 mm to approximate the diffuser.
