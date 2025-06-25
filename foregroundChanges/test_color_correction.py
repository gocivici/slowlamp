import numpy as np
import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os

from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
import correct_color_RGBW
import board
import neopixel

neopixels = neopixel.NeoPixel(board.D12, 24, brightness=0.2, pixel_order = neopixel.GRBW)
    

for r in list(range(0, 255, 64))+[255]:
	for g in list(range(0, 255, 64))+[255]:
		for b in list(range(0, 255, 64))+[255]:
			#target_colors.append( (r, g, b))
			target_rgb = (r, g, b)
			print("target_rgb", target_rgb)
			
			correct_rgbw =  correct_color_RGBW.correct_color(target_rgb)
			print("correct_rgbw", correct_rgbw)

			neopixels.fill(correct_rgbw)

			canvas = np.ones((500, 500, 3)).astype(np.uint8)
			canvas[:, :, 0] = target_rgb[2]
			canvas[:, :, 1] = target_rgb[1]
			canvas[:, :, 2] = target_rgb[0]

			cv2.imshow('frame', canvas)

			cv2.waitKey(0)

			neopixels.fill(target_rgb)

			res = cv2.waitKey(0)
			
			if res == ord('q'):
				break
				
		if res == ord('q'):
			break
			
	if res == ord('q'):
		break
				
neopixels.fill((0, 0, 0, 0))
