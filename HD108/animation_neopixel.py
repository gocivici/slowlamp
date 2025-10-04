import board
import neopixel
import colour 
import numpy as np
import spidev
import time

frames = []

start_color_rgb = (164/255, 0, 214/255)
end_color_rgb = (255/255, 255/255, 0)

total_steps = 255
num_pixels = 4

pixels = neopixel.NeoPixel(board.D18, num_pixels, brighness=0.2)

start_color_oklab = colour.convert(start_color_rgb, "sRGB", "Oklab")
end_color_oklab = colour.convert(end_color_rgb, "sRGB", "Oklab")

start_color = np.array(start_color_oklab)
end_color = np.array(end_color_oklab)

for step in range(total_steps):

    frame = []
    position = step/total_steps*num_pixels
    for pixel in range(num_pixels):

        if pixel < position:
            color_step = (end_color - start_color)/position * pixel + start_color
        else:
            color_step = (start_color - end_color)/(num_pixels - position) * (pixel - position) + end_color
            
        srgb_color_step = colour.convert(color_step, "Oklab", "sRGB")
        srgb = (np.array(srgb_color_step) * (2**16 - 1)).astype(np.uint16)
        frame.append(tuple(srgb))
    # print(srgb)
    frames.append(frame)

def send_neopixel_colors(colors_16bit):
    

    
    for r16, g16, b16 in list(colors_16bit):
        r16, g16, b16 = map(int, (r16, g16, b16))
        
        pixels[1] = (int(r16/256), int(g16/256), int(b16/256))
        # can't figure out how to loop over te position of the LEDs
        





#pixels[0] = (255, 0, 0)

total_time = 3 #seconds
time_sleep = total_time/total_steps

for frame in frames:
    send_neopixel_colors(frame)
    
    #exit()
    time.sleep(time_sleep)


