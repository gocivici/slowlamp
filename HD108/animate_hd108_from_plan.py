import numpy as np

import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)                                            # Open SPI bus 0, device 0
spi.max_speed_hz = 15000000                               # Set speed to 10MHz: 32000000 maximum

def send_hd108_colors_with_brightness(colors_16bit):
    
    data = []
    data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
    
    for brightness, r16, g16, b16 in list(colors_16bit):
        brightness, r16, g16, b16 = map(int, (brightness, r16, g16, b16))
        
        # print(int(r16/255), int(g16/255), int(b16/255))
        if brightness <= 0:
            brightness_frame = (1 << 15) | (1 << 10) | (1 << 5) | 1
        else:
            brightness_frame = (1 << 15) | (brightness << 10) | (brightness << 5) | brightness
        
        data.extend([
            (brightness_frame >> 8) & 0xFF, brightness_frame & 0xFF,             # Brightness
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
        
    num_end = 2 * (len(colors_16bit) + 1) 
    data.extend([0xFF] * num_end)  # End frame
    spi.writebytes(data)


animation_plan = np.load("animation_plan.npz")["animation_plan"]
fps = 15


for f in animation_plan:
    frame = f[1:].reshape(-1, 4)/3
    num_leds = len(frame)
    send_hd108_colors_with_brightness(frame)
    time.sleep(1/fps)

# np.savez("animation_plan.npz", animation_plan=animation_plan)

colors_16bit = [(1, 0, 0, 0)]*num_leds
send_hd108_colors_with_brightness(colors_16bit)
send_hd108_colors_with_brightness(colors_16bit)
spi.close()