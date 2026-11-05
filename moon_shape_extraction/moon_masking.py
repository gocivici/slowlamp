import numpy as np
import os
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

def get_led_points_pcb():
    index_at_ring = [72, 136, 196, 248, 296, 336, 368, 396, 416, 428, 434, 435]
    max_ring = 220 #d = mm
    min_ring = 20 #d = mm
    led_points = []
    for i in range(len(index_at_ring)):
        current_num = index_at_ring[i]
        if i > 0:
            current_num -= index_at_ring[i-1]
        current_r = (max_ring/2-min_ring/2)/(len(index_at_ring)-2)*(len(index_at_ring)-2-i) + min_ring/2
        # len = 12, i = 11, r = 100/10*(-1)+10 = 0
        # len = 12, i = 0, r = 100/10*(10)+10 = 110 
        for l in range(current_num):
            x = - current_r*np.sin(np.pi*2/current_num*l)
            y = - current_r*np.cos(np.pi*2/current_num*l)
            led_points.append([x, y])
    print(f"initialized lamp with {len(led_points)} leds. ")
    return np.array(led_points)

def get_gif_frames():
    # Open the GIF file
    with Image.open("moon_shape_extraction/Moon_movie_small.gif") as img:
        # Convert every frame to RGBA or RGB and bundle into a single numpy array
        gif_array = np.array([np.array(frame.convert("L")) for frame in ImageSequence.Iterator(img)])

    print(gif_array.shape)
    print(gif_array[0, 0, 0])
    return gif_array

led_points = get_led_points_pcb()
led_min_x = np.min(led_points[:, 0])
led_max_x = np.max(led_points[:, 0])    
led_min_y = np.min(led_points[:, 1])
led_max_y = np.max(led_points[:, 1])

num_leds = led_points.shape[0]
frames = get_gif_frames()  
# Skip the first frame since it's a repeat and skip 14th frame since there is a bug
frames = np.delete(frames, 0, axis=0)
frames = np.delete(frames, 13, axis=0)

moon_frames = [[[2, 2**16-1, 2**16-1, 2**16-1]]*num_leds for _ in range(frames.shape[0])]
illuminated_pixels = []

full_moon_size = 0
for fi, f in enumerate(frames):
    print(f"Processing frame {fi}/{frames.shape[0]}")
    # mask = np.where(f > 127, True, False)
    true_points_grid = np.argwhere(f > 10)
    false_points_grid = np.argwhere(f <= 10)
    illuminated_pixels.append(true_points_grid.shape[0])
    print(true_points_grid.shape)
    if full_moon_size < true_points_grid.shape[0]:
        full_moon_size = true_points_grid.shape[0]
    # compare if led_points are closer to true_points_grid than false_points_grid

    if true_points_grid.shape[0] == 0:
        print(f"Frame {fi} has no illuminated pixels.")
        continue

    moon_min_y = np.min(true_points_grid[:, 0])
    moon_max_y = np.max(true_points_grid[:, 0])
    print("moon_range", moon_min_y, moon_max_y)
    transformed_led_points = led_points.copy()
    transformed_led_points[:, 0] = (transformed_led_points[:, 0] - led_min_x) / (led_max_x - led_min_x) * (moon_max_y - moon_min_y) + moon_min_y
    transformed_led_points[:, 1] = (transformed_led_points[:, 1] - led_min_y) / (led_max_y - led_min_y) * (moon_max_y - moon_min_y) + moon_min_y

    for li, led in enumerate(transformed_led_points):
        led = np.array(led)
        distances_to_true = np.linalg.norm(true_points_grid - led, axis=1)
        distances_to_false = np.linalg.norm(false_points_grid - led, axis=1)
        if np.min(distances_to_true) < np.min(distances_to_false):
            # print(f"led {led} is closer to true points than false points")
            pass
        else:
            # print(f"led {led} is closer to false points than true points")
            moon_frames[fi][li] = [2, 0, 0, 0]
            pass

    # plot the frame with the lit leds
    plt.figure(figsize=(10, 10))
    plt.imshow(f, cmap='jet')
    led_colors = np.array(moon_frames[fi])[:, 1:]/2**16
    plt.scatter(transformed_led_points[:, 1], transformed_led_points[:, 0], c=led_colors, s=50, alpha=0.7)
    plt.show()


print(f"Full moon size: {full_moon_size}")
illumination_ratios = np.array([illuminated_pixels[i]/full_moon_size for i in range(len(illuminated_pixels))])
print(f"Illumination ratios: {illumination_ratios}")
# trend of waning and waxing moon
trend = (illumination_ratios[1:] - illumination_ratios[:-1])>0
print(f"Trend of moon illumination: {trend}")

np.savez("moon_shape_extraction/moon_frames.npz", moon_frames = np.array(moon_frames), 
        illumination_ratios = np.array(illumination_ratios), trend = np.array(trend))
