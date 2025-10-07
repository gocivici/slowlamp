import numpy as np
import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os

from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color

import spidev

num_leds = 11

import board
import neopixel

pixels = neopixel.NeoPixel(board.D18, num_leds, brightness=0.3, pixel_order = neopixel.GRBW)

spi = spidev.SpiDev()
spi.open(0, 0)                                            # Open SPI bus 0, device 0
spi.max_speed_hz = 2000000                               # Set speed to 10MHz: 32000000 maximum
    
        
def send_hd108_colors(colors_16bit, global_brightness=5):
    
    data = []
    data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
    
    for r16, g16, b16 in list(colors_16bit):
        r16, g16, b16 = map(int, (r16, g16, b16))
        
        # print(int(r16/255), int(g16/255), int(b16/255))
        if global_brightness <= 0:
            brightness = 0 #gives werid colors
        else:
            brightness = (1 << 15) | (global_brightness << 10) | (global_brightness << 5) | global_brightness
        
        data.extend([
            (brightness >> 8) & 0xFF, brightness & 0xFF,             # Brightness
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
        
    num_end = 2 * (len(colors_16bit) + 1) 
    data.extend([0xFF] * num_end)  # End frame
    spi.writebytes(data)
    
    
for r in list(range(0, 255, 64))+[255]:
	for g in list(range(0, 255, 64))+[255]:
		for b in list(range(0, 255, 64))+[255]:
			#target_colors.append( (r, g, b))
			target_rgb = (r, g, b)
			print("target_rgb", target_rgb)
            
			colors_16bit = [(r*256, g*256, b*256)]*num_leds
			send_hd108_colors(colors_16bit)
			send_hd108_colors(colors_16bit)

			# time.sleep(1)
			neo_color = (r, g, b, 0)
			pixels.fill(neo_color)
             
			canvas = np.ones((500, 500, 3)).astype(np.uint8)
			canvas[:, :, 0] = target_rgb[2]
			canvas[:, :, 1] = target_rgb[1]
			canvas[:, :, 2] = target_rgb[0]

			cv2.imshow('frame', canvas)

			res = cv2.waitKey(0)

			# neopixels.fill(target_rgb)

			# res = cv2.waitKey(0)
			
			if res == ord('q'):
				break
				
		if res == ord('q'):
			break
			
	if res == ord('q'):
		break
				
colors_16bit = [(0, 0, 0)]*num_leds
send_hd108_colors(colors_16bit, global_brightness = 0)
send_hd108_colors(colors_16bit, global_brightness = 0)
pixels.fill((0,0,0,0))
spi.close()
