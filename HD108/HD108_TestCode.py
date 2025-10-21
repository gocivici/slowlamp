import spidev
import time

def send_hd108_colors(colors_16bit, global_brightness=15):
    spi = spidev.SpiDev()
    spi.open(0, 0)                                            # Open SPI bus 0, device 0
    spi.max_speed_hz = 2000000                               #2M # Set speed to 10MHz: 32000000 maximum
    
    data = []
    data.extend([0x00] * 8)  #16 # Start frame: HD108 protocol requires 128-bit of zeros start frame before LED data
    
    for r16, g16, b16 in colors_16bit:
        # 5-bit brightness (0-31)
        brightness_5bit = min(global_brightness, 31)          # Loops through each LED's RGB values (0-65535 range) Ensures brightness stays within valid 5-bit range (0-31)
        
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

# Test code
test_colors = [
    (65535, 0, 0), (0, 65535, 0), (0, 0, 65535),
    (65535, 65535, 0), (32767, 32767, 32767),
    (65535, 0, 0), (0, 65535, 0), (0, 0, 65535),
    (65535, 65535, 0), (32767, 32767, 32767),
    (65535, 0, 0)
]
# send_hd108_colors(test_colors, global_brightness=31)
# time.sleep(2)

off_colors = [(100, 100, 100) for _ in range(len(test_colors))]
send_hd108_colors(off_colors, global_brightness=1)
send_hd108_colors(off_colors, global_brightness=1)
time.sleep(2)
