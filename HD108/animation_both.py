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
num_pixels = 11

pixels = neopixel.NeoPixel(board.D18, num_pixels, brightness=0.2, pixel_order = neopixel.GRBW)

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
    

    
    for i, (r16, g16, b16) in enumerate(list(colors_16bit)):
        r16, g16, b16 = map(int, (r16, g16, b16))
        
        pixels[i] = (int(r16/256), int(g16/256), int(b16/256), 0)
        # can't figure out how to loop over te position of the LEDs
        


def send_hd108_colors(colors_16bit, global_brightness=1):
    spi = spidev.SpiDev()
    spi.open(0, 0)                                            # Open SPI bus 0, device 0
    spi.max_speed_hz = 2000000                               # Set speed to 10MHz: 32000000 maximum
    
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
    spi.close()


#pixels[0] = (255, 0, 0)

total_time = 3 #seconds
time_sleep = total_time/total_steps

for frame in frames:
    send_hd108_colors(frame)
    send_neopixel_colors(frame)
    
    #exit()
    time.sleep(time_sleep)

for frame in frames:
    send_hd108_colors(frame)
    send_neopixel_colors(frame)
    
    #exit()
    time.sleep(time_sleep)

for frame in frames:
    send_hd108_colors(frame)
    send_neopixel_colors(frame)
    
    #exit()
    time.sleep(time_sleep)

for frame in frames:
    send_hd108_colors(frame)
    send_neopixel_colors(frame)
    
    #exit()
    time.sleep(time_sleep)


