import colour 
import numpy as np
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)                                            # Open SPI bus 0, device 0
spi.max_speed_hz = 15000000                               # Set speed to 10MHz: 32000000 maximum

start_color_rgb = (164/255, 0, 214/255)
end_color_rgb = (255/255, 255/255, 0)

frames = []
total_steps = 255
num_pixels = 11

start_color_oklab = colour.convert(start_color_rgb, "sRGB", "Oklab")
end_color_oklab = colour.convert(end_color_rgb, "sRGB", "Oklab")

start_color = np.array(start_color_oklab)
end_color = np.array(end_color_oklab)


for step in range(total_steps):
    # start_color_step = (end_color - start_color)/total_steps * step + start_color
    # end_color_step = end_color - (end_color - start_color)/total_steps * step
    frame = []
    position = step/total_steps*num_pixels
    for pixel in range(num_pixels):
        # color_step = (end_color_step - start_color_step)/num_pixels * pixel + start_color_step
        if pixel < position:
            color_step = (end_color - start_color)/position * pixel + start_color
        else:
            color_step = (start_color - end_color)/(num_pixels - position) * (pixel - position) + end_color
            
        srgb_color_step = colour.convert(color_step, "Oklab", "sRGB")
        srgb = (np.array(srgb_color_step) * (2**16 - 1)).astype(np.uint16)
        frame.append(tuple(srgb))
    # print(srgb)
    frames.append(frame)


def send_hd108_colors(colors_16bit, global_brightness=15):
    
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

total_time = 3 #seconds
time_sleep = total_time/total_steps

for frame in frames:
    # print(frame)
    send_hd108_colors(frame, global_brightness = 5)
    #exit()
    time.sleep(time_sleep)

for frame in frames:
    # print(frame)
    send_hd108_colors(frame, global_brightness = 5)
    #exit()
    time.sleep(time_sleep)
    
for frame in frames:
    # print(frame)
    send_hd108_colors(frame, global_brightness = 5)
    #exit()
    time.sleep(time_sleep)

spi.close()
