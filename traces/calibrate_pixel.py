import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import copy
import color_functions
from colormath.color_diff import delta_e_cie2000 #numpy error asscalar 
                # | ∆E$_{00}$ Value | Perceptual Difference                        |
                # | --------------- | -------------------------------------------- |
                # | 0–1             | **Not perceptible** by human eyes            |
                # | 1–2             | **Perceptible through close observation**    |
                # | 2–10            | **Noticeable at a glance**                   |
                # | 11–49           | **Colors are more different than similar**   |
                # | 50+             | **Perceived as completely different colors** |

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment
import time

using_pi = True
picam2 = None
if using_pi:
    import board
    import neopixel
    # pixels = neopixel.NeoPixel(board.D12, 1, brightness=125)
    pixels = neopixel.NeoPixel(board.D12, 24, brightness=0.5, pixel_order = neopixel.GRBW)

    from picamera2 import Picamera2
    # picamera2 needs numpy 1 seems like
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280,720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")

    picam2.set_controls({'AwbEnable': False})
    picam2.set_controls({'AeEnable': False})
    picam2.set_controls({'AnalogueGain': 15.0, 'ExposureTime': 10000})

    picam2.start()


pixel_colors = []
# for r in range(0, 255, 10):
# 	for g in range(0, 255, 10):
# 		for b in range(0, 255, 10):
# 			pixel_colors.append( (r, g, b))

pixel_colors.append( (0, 0, 0))

pixel_colors.append( (255, 0, 0))
pixel_colors.append( (0, 255, 0))
pixel_colors.append( (0, 0, 255))

pixel_colors.append( (255, 255, 0))
pixel_colors.append( (0, 255, 255))
pixel_colors.append( (255, 0, 255))

pixel_colors.append( (255, 255, 255))

for r in range(0, 255, 32):
	for g in range(0, 255, 32):
		for b in range(0, 255, 32):
			pixel_colors.append( (r, g, b))

with_white = []
for color in pixel_colors:
    for w in list(range(0, 255, 32))+[255]:
        with_white.append((*color, w))

result_colors = []

canvas_size = 500
feed_preview_size = 100
color_swatch_size = 50

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text


max_num_colors = 5
color_max_distance = 12

trace_length_min = 3
trace_length_max = 6
max_frame_number = 10


def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)


def lab_to_rgb(lab):
    # Create LabColor object
    lab_color = LabColor(*lab)
    # Convert to sRGB
    rgb_color = convert_color(lab_color, sRGBColor)
    
    # Convert to 0-255 integers (clip to valid range)
    rgb_clipped = np.clip([rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b], 0, 1)
    return (rgb_clipped * 255).astype(int)

def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

def extract_center_color(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    crop = frame[cy-15:cy+15, cx-15:cx+15]
    avg_color = crop.mean(axis=(0, 1))  # BGR
    return avg_color[::-1]  # Return as RGB

cap = None
if not using_pi:
    # Initialize the camera module
    cap = cv2.VideoCapture(0)

    # Check if the camera is initialized correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

tracks = [] #rgb
canvas_color = None

test_index = 0

# Display the camera feed
while test_index < len(with_white):
    test_color = with_white[test_index]
    if using_pi:
        pixels.fill(test_color)
    
    time.sleep(0.6)

    test_index += 1

    if using_pi:
        frame = picam2.capture_array()
    else:
        ret, frame = cap.read()

    # cv2.imshow('frame', frame)
    recevied_color = extract_center_color(frame) #rgb
    result_colors.append(recevied_color)

    canvas = np.ones((500, 500, 3)).astype(np.uint8)
    if canvas_color is not None:
        # canvas_rgb_color = canvas_color #lab_to_rgb(canvas_color)
        canvas[:, :, 0] = recevied_color[2]
        canvas[:, :, 1] = recevied_color[1]
        canvas[:, :, 2] = recevied_color[0]
    feed_preview = resize_to_max(frame, feed_preview_size)
    canvas[0:feed_preview.shape[0], 0:feed_preview.shape[1]] = feed_preview
    feed_preview_y = feed_preview.shape[0]
    feed_preview_x = feed_preview.shape[1]
    canvas = cv2.putText(canvas, f'pixel (rgbw): ({test_color[0]}, {test_color[1]}, {test_color[2]}, {test_color[3]})', (feed_preview_x, 20), font, font_scale, text_color, thickness)
    canvas = cv2.putText(canvas, f'camera (rgb): ({int(recevied_color[0])}, {int(recevied_color[1])}, {int(recevied_color[2])})', (feed_preview_x, 50), font, font_scale, text_color, thickness)
        

    cv2.imshow('frame', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if not using_pi:
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()
else:
    pixels.fill((0, 0, 0, 0))

np.savez("pairings-32-GRBW.npz", pixel_colors = with_white, result_colors = result_colors)
