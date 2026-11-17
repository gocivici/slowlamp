"""
Process-pool job: one transition between two keyframes, through the snapshot's
interpolate_two_frames. Kept in its own module so worker processes import only this.
"""

import io
import os
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_snapshot"))

with redirect_stdout(io.StringIO()):
    import helper_classes
    import fgc_arnab_animation as anim

# Max swatch samples per transition (the swatch view doesn't need every LED frame).
SWATCH_SAMPLES = 240


def to_drive(plan):
    """Animation plan rows -> exactly what send_hd108_colors_with_brightness puts on SPI:
    map(int, ...) truncation, brightness <= 0 sent as 1. Shape (frames, leds, 4) uint16."""
    data = plan[:, 1:].reshape(len(plan), -1, 4)
    drive = np.trunc(data)
    b = drive[:, :, 0]
    drive[:, :, 0] = np.where(b <= 0, 1, b)
    return np.clip(drive, 0, 2**16 - 1).astype("<u2")


def swatch_frames(rec_a, rec_b, n):
    """Cross-fade the 5 record colours in OKLab, converted back with the lamp's own
    gamut mapping (oklab_to_srgb_255). For the swatch view only."""
    samples = max(2, min(n, SWATCH_SAMPLES))
    ca = helper_classes.sRGB_to_oklab_vec(np.array(rec_a["colors"], dtype=float) / 255)
    cb = helper_classes.sRGB_to_oklab_vec(np.array(rec_b["colors"], dtype=float) / 255)
    w = np.linspace(0, 1, samples)[:, None, None]
    lab = (1 - w) * ca[None] + w * cb[None]
    rgb = helper_classes.oklab_to_srgb_255(lab.reshape(-1, 3)).reshape(samples, 5, 3)
    counts = (1 - w[:, :, 0]) * np.array(rec_a["counts"]) + w[:, :, 0] * np.array(rec_b["counts"])
    return {
        "colors": np.clip(np.round(rgb), 0, 255).astype(int).tolist(),
        "counts": np.round(counts).astype(int).tolist(),
    }


def interpolate_two_frames_batched(key_frame1, key_frame2, animation_length):
    """fgc_arnab.interpolate_two_frames with the per-LED colour conversion done in one
    call instead of one call per LED. Same interp, same oklab_to_srgb_255, same clip and
    round; check_batched.py compares it against the original on real records."""
    if animation_length <= 0:
        return np.zeros((1, len(key_frame1)))

    key_frames = np.stack([key_frame1, key_frame2], axis=0)
    frame_indices = key_frames[:, 0]
    animation_plan = np.zeros((animation_length, len(key_frame1)))
    time_arr = np.linspace(frame_indices[0], frame_indices[-1] + 1, animation_length)

    leds = key_frames[:, 1:].reshape(2, -1, 4)  # (keyframe, led, [watt, L, a, b])
    n = leds.shape[1]
    oklab_full = np.empty((animation_length, n, 3))
    brightness = np.empty((animation_length, n))
    for j in range(n):
        for c in range(3):
            oklab_full[:, j, c] = helper_classes.interpolate_stretched(frame_indices, leds[:, j, c + 1], time_arr)
        input_brightness = np.clip((leds[:, j, 0] / anim.base_watt) * 10, 1, 6)
        brightness[:, j] = helper_classes.interpolate_stretched(frame_indices, input_brightness, time_arr)

    rgb_full = helper_classes.oklab_to_srgb_255(oklab_full.reshape(-1, 3)).reshape(animation_length, n, 3)
    input_pixels = np.clip(rgb_full * 255, 0, 2**16 - 1).round()

    out = animation_plan[:, 1:].reshape(animation_length, n, 4)
    out[:, :, 0] = brightness
    out[:, :, 1:] = input_pixels
    return animation_plan


def transition_job(start_keyframe, target_keyframe, animation_length, rec_a, rec_b):
    with redirect_stdout(io.StringIO()):
        plan = interpolate_two_frames_batched(start_keyframe, target_keyframe, animation_length)
        return to_drive(plan).tobytes(), swatch_frames(rec_a, rec_b, len(plan))


def hold_job(start_keyframe, rec):
    """Single record: the lamp shows the start keyframe via interpolate_two_frames(start, start, 3)[0]."""
    with redirect_stdout(io.StringIO()):
        plan = anim.interpolate_two_frames(start_keyframe, start_keyframe, 3)[:1]
        return to_drive(plan).tobytes(), swatch_frames(rec, rec, 1)
