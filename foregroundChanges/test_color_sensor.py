# SPDX-FileCopyrightText: 2020 Bryan Siepert, written for Adafruit Industries
# SPDX-License-Identifier: MIT
from time import sleep

import board

from adafruit_as7341 import AS7341

import colour
from colormath.color_objects import XYZColor, sRGBColor
from colormath.color_conversions import convert_color

import numpy as np
import cv2

i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller
sensor = AS7341(i2c)

import board
import neopixel

neopixels = neopixel.NeoPixel(board.D12, 24, brightness=1, pixel_order = neopixel.GRBW)

neopixels.fill((255, 70, 98))

# observer_start = 360 #nm
# observer_end = 830 #nm

sensor_wavelengths = np.array([415, 445, 480, 515, 555, 590, 630, 680])
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'] 

cmf_X = cmfs.values[:, 0]
cmf_Y = cmfs.values[:, 1]
cmf_Z = cmfs.values[:, 2]

wavelengths = cmfs.wavelengths

X_bar = np.interp(sensor_wavelengths, wavelengths, cmf_X)
Y_bar = np.interp(sensor_wavelengths, wavelengths, cmf_Y)
Z_bar = np.interp(sensor_wavelengths, wavelengths, cmf_Z)




def bar_graph(read_value):
    scaled = int(read_value / 1000)
    return "[%5d] " % read_value + (scaled * "*")


while True:
    print("F1 - 415nm/Violet  %s" % bar_graph(sensor.channel_415nm))
    print("F2 - 445nm//Indigo %s" % bar_graph(sensor.channel_445nm))
    print("F3 - 480nm//Blue   %s" % bar_graph(sensor.channel_480nm))
    print("F4 - 515nm//Cyan   %s" % bar_graph(sensor.channel_515nm))
    print("F5 - 555nm/Green   %s" % bar_graph(sensor.channel_555nm))
    print("F6 - 590nm/Yellow  %s" % bar_graph(sensor.channel_590nm))
    print("F7 - 630nm/Orange  %s" % bar_graph(sensor.channel_630nm))
    print("F8 - 680nm/Red     %s" % bar_graph(sensor.channel_680nm))
    print("Clear              %s" % bar_graph(sensor.channel_clear))
    print("Near-IR (NIR)      %s" % bar_graph(sensor.channel_nir))
    print("\n------------------------------------------------")
    sensor_readings = [sensor.channel_415nm, sensor.channel_445nm, sensor.channel_480nm, sensor.channel_515nm,
    sensor.channel_555nm, sensor.channel_590nm, sensor.channel_630nm, sensor.channel_680nm]
    
    I = np.array(sensor_readings)
    I = I/(2**16)
    
    X = np.sum(I * X_bar)
    Y = np.sum(I * Y_bar)
    Z = np.sum(I * Z_bar)
    
    xyz_color = XYZColor(X, Y, Z)
    print("sensed xyz", X, Y, Z)
    
    rgb_color = convert_color(xyz_color, sRGBColor)
    r, g, b = rgb_color.get_upscaled_value_tuple()
    
    if r > 255 or g > 255 or b > 255:
        max_val = max(r, max(g, b))
        r = int(r*255/max_val)
        g = int(g*255/max_val)
        b = int(b*255/max_val)
		
    print("sensed rgb", r, g, b)
    
    canvas = np.ones((500, 500, 3)).astype(np.uint8)
    canvas[:, :, 0] = b
    canvas[:, :, 1] = g
    canvas[:, :, 2] = r
	
    cv2.imshow('frame', canvas)
    
    res = cv2.waitKey(0)
	
    if res == ord('q'):
        break
				
    sleep(1)

neopixels.fill((0, 0, 0))
