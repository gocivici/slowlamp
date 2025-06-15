import board
import neopixel
import time
pixels = neopixel.NeoPixel(board.D12, 24, bpp = 4, brightness = 0.1, pixel_order = neopixel.GRBW)

pixels[0] = (255, 127, 0, 0)
pixels.fill((255, 0, 0, 0))

for r in range(0, 255, 30):
	for g in range(0, 255, 30):
		for b in range(0, 255, 30):
			pixels[0] = (r, g, b, 0)
			time.sleep(1)
