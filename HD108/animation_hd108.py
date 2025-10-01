import colour 
import numpy as np
import spidev
import time

start_color_rgb = (164, 0, 214)
end_color_rgb = (255, 255, 0)

frames = []
total_steps = 255
num_pixels = 11

# Convert sRGB to Oklab
srgb_color = [0.5, 0.2, 0.8]  # Example sRGB color
oklab_color = colour.convert(srgb_color, "sRGB", "Oklab")
print(oklab_color)

# Convert Oklab to sRGB
oklab_color_back = [0.4, 0.1, -0.3] # Example Oklab color
srgb_color_back = colour.convert(oklab_color_back, "Oklab", "sRGB")
print(srgb_color_back)

start_color_oklab = colour.convert(start_color_rgb, "sRGB", "Oklab")
end_color_oklab = colour.convert(end_color_rgb, "sRGB", "Oklab")

start_color = np.array(start_color_oklab)
end_color = np.array(end_color_oklab)


for step in range(total_steps):
    start_color_step = (end_color - start_color)/total_steps * step + start_color
    end_color_step = end_color - (end_color - start_color)/total_steps * step
    frame = []
    for pixel in range(num_pixels):
        color_step = (end_color_step - start_color_step)/num_pixels * pixel + start_color_step
        srgb_color_step = colour.convert(color_step, "Oklab", "sRGB")
        srgb = int(np.array(srgb_color_step)*2**16).astype(np.int16)
        frame.append(srgb)
    frames.append(frame)


def send_hd108_colors(colors_16bit, global_brightness=15):
    spi = spidev.SpiDev()
    spi.open(0, 0)                                            # Open SPI bus 0, device 0
    spi.max_speed_hz = 20000000                               # Set speed to 10MHz: 32000000 maximum
    
    data = []
    data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
    
    for r16, g16, b16 in colors_16bit:

        brightness = (1 << 15) | (global_brightness << 10) | (global_brightness << 5) | global_brightness
        
        data.extend([
            (brightness >> 8) & 0xFF, brightness & 0xFF,             # Brightness
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
    
    data.extend([0xFF] * len(colors_16bit))  # End frame
    spi.writebytes2(data)
    spi.close()

total_time = 15 #seconds
time_sleep = total_time/total_steps

for frame in frames:
    send_hd108_colors(frame, 15)
    time.sleep(time_sleep)

