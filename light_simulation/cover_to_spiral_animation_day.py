import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
import cover_sim
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

cover_img = "archive-sim-correct1.png"
data = cover_sim.retrieve(cover_img)

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

def average_color_block(colors):
    colors = colors.reshape(-1, 3)
    return np.average(colors, axis=0)

def get_chroma(rgb): 
    # how saturated or vivid one color is 
    rgb_scaled = sRGBColor(*[v / 255 for v in rgb], is_upscaled=False)
    lab = convert_color(rgb_scaled, LabColor)
    return (lab.lab_a ** 2 + lab.lab_b ** 2) ** 0.5

def most_vib_color(colors):
    vr_chosen_id = 0
    vr_count = 0
    max_chromma = 0
    colors = colors.reshape(-1, 3)
    for i, color in enumerate(colors):
        chroma = get_chroma(color)
        if chroma >= max_chromma:
            max_chromma = chroma
            vr_chosen_id = i
    return colors[vr_chosen_id]

def largest_color(colors, counts):
    colors = colors.reshape(-1, 3)
    counts = counts.reshape(-1, 1)
    return colors[np.argmax(counts)]

main_colors = []
vibrant_colors = []
counts = []
hours = []

for record in data:
    # Extract main color using regex
    main_colors.append([record["vc"], record["ac1"], record["ac2"], record["ac3"], record["ac4"]])
    counts.append([record["vc_px"], record["ac1_px"], record["ac2_px"], record["ac3_px"], record["ac4_px"]])
    vibrant_colors.append(record["vc"])
    hours.append(record["timestamp"])

print(main_colors)
print(vibrant_colors)
# print(days)

# glowing_canvas = np.zeros((img_size, img_size, 3))

main_colors_array = np.array(main_colors)
print(main_colors_array[0])
print(main_colors_array.shape)
main_colors_array = main_colors_array.reshape(-1, 24, 5, 3)
print(main_colors_array.shape)
print(main_colors_array[0, 0])

counts_array = np.array(counts)
counts_array = counts_array.reshape(-1, 24, 5)

for end_i in range(5, len(main_colors_array), 2): #len(days)-1):
    # end_day = end_i
    data_length = end_i

    start_i = 0
    img_size = 900

    canvas = np.ones((img_size, img_size, 3)).astype(np.uint8) *255

    circle_center = (img_size // 2, img_size // 2)
    ambient_radius = 350
    angle_per_swatch = 360/24

    center_x, center_y = img_size // 2, img_size // 2
    a = 20  # Controls initial radius

    max_angle = max(2*np.pi, ((end_i- start_i) * angle_per_swatch)/180*np.pi)
    # print("max_angle", max_angle/np.pi*180)

    b = (ambient_radius - a) / max_angle *0.6   # Controls how fast the radius increases
    thickness = b * 2 * np.pi
    # print("ring thickness", thickness) 

    for j in range(end_i):

        
        # selected_color = to_cv2_color(most_vib_color(main_colors_array[end_i-j]))
        selected_color = to_cv2_color(largest_color(main_colors_array[end_i-j], counts_array[end_i-j]))
        # trace_vibrant_color = to_cv2_color(vibrant_colors[end_i-j])
        
        start_angle = ((end_i-(j-start_i))//24*360) + hours[end_i-j] * angle_per_swatch - 90
        # print(start_angle)
        radius = int(a + b * ((start_angle + angle_per_swatch/2 + 90)/180*np.pi))
        x = int(center_x + radius * np.cos(start_angle + angle_per_swatch/2))
        y = int(center_y + radius * np.sin(start_angle + angle_per_swatch/2))
        # points.append((x, y))
        # print(radius)

        canvas = cv2.ellipse(canvas, circle_center,(radius, radius), start_angle, 0, angle_per_swatch,
                                selected_color, -1)

    # largest_color = main_colors[data_length]
    # vibrant_color = vibrant_colors[data_length]
    # current_ambient_color = to_cv2_color(largest_color)
    # current_vibrant_color =to_cv2_color(vibrant_color)

    # vr_percent = 300/(1000+500)

    # canvas = cv2.circle(canvas, circle_center, a, current_ambient_color, -1)
    # canvas = cv2.circle(canvas, circle_center, int((a)*vr_percent**(0.5)), current_vibrant_color, -1)

    cv2.imwrite(f"C:/work/slow_lamp/render_test/largest_animation/render_spiral_{end_i:05d}.png", canvas)