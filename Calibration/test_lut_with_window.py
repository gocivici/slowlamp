import numpy as np
import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os

from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
# import correct_color_RGBW
import board
import spidev
import json


with open("slow_lamp_calibration.json") as f:
    cal = json.load(f)


# --- Option A: full LUT, direct lookup, ~1.5MB in memory ---

r_full = cal["luts_full"]["red"]
g_full = cal["luts_full"]["green"]
b_full = cal["luts_full"]["blue"]


def correct_full(r, g, b):
    """Direct lookup, no math at runtime."""
    return r_full[r], g_full[g], b_full[b]

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

for r in list(range(0, 255, 64))+[255]:
	for g in list(range(0, 255, 64))+[255]:
		for b in list(range(0, 255, 64))+[255]:
			#target_colors.append( (r, g, b))
			target_rgb = (r, g, b)
			print("target_rgb", target_rgb)
			
			# correct_rgbw =  correct_color_RGBW.correct_color(target_rgb)
			# print("correct_rgbw", correct_rgbw)

			fill_all(target_rgb[0]*255, target_rgb[1]*255, target_rgb[2]*255)

			canvas = np.ones((500, 500, 3)).astype(np.uint8)
			canvas[:, :, 0] = target_rgb[2]
			canvas[:, :, 1] = target_rgb[1]
			canvas[:, :, 2] = target_rgb[0]

			cv2.imshow('frame', canvas)

			res = cv2.waitKey(0)

			fill_all(correct_full(target_rgb[0]*255, target_rgb[1]*255, target_rgb[2]*255))

			res = cv2.waitKey(0)
			
			if res == ord('q'):
				break
				
		if res == ord('q'):
			break
			
	if res == ord('q'):
		break
				
fill_all(0, 0, 0)
