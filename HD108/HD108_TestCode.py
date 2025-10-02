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
        
        # 16-bit brightness field: 1 + 5 bits brightness + 10 bits padding
        byte1 = 0x80 | (brightness_5bit << 2)
        byte2 = 0x00
        # as (1)(5bit)(5bit)(5bit) brightnesses
        
        
        data.extend([
            0xFF, 0xFF,                              # Brightness max first
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
    
    # data.extend([0xFF] * 8)  # End frame #? 
    num_end = 2 * (len(colors_16bit) + 1) 
    print(num_end) 
    data.extend([0xFF] * num_end)
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
send_hd108_colors(test_colors, global_brightness=31)
time.sleep(2)

off_colors = [(0,0,0) for _ in range(len(test_colors))]
send_hd108_colors(off_colors, global_brightness=0)
