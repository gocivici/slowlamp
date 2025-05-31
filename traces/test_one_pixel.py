import board
import neopixel
import time
pixels = neopixel.NeoPixel(board.D12, 1, brightness = 125)

pixels[0] = (255, 127, 0)

for r in range(0, 255, 10):
	for g in range(0, 255, 10):
		for b in range(0, 255, 10):
			pixels[0] = (r, g, b)
			time.sleep(1)
