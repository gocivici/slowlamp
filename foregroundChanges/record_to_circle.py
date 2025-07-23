import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re

record = "C:/work/slow_lamp/july_5_days.txt"

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

main_colors = []
vibrant_colors = []
hours = []

with open(record, 'r') as file:
    for line in file:
        # Extract main color using regex
        main_match = re.search(r"main_color \(rgb\): \((\d+), (\d+), (\d+)\)", line)
        if main_match:
            main_rgb = tuple(map(int, main_match.groups()))
            main_colors.append(main_rgb)

        # Extract supplemental colors using regex
        supp_match = re.search(r"supplemental_colors: \[(.*?)\]", line)
        if supp_match:
            color_list = supp_match.group(1)
            # Extract all RGB tuples from the list
            all_supp_colors = re.findall(r"\((\d+), (\d+), (\d+)\)", color_list)
            if len(all_supp_colors) >= 2:
                second_rgb = tuple(map(int, all_supp_colors[1]))
                vibrant_colors.append(second_rgb)

        time_match = re.search(r"@ \d{4}-\d{2}-\d{2} (\d{2}):\d{2}:\d{2}", line)
        if time_match:
            hour = int(time_match.group(1))
            hours.append(hour)


print(main_colors)
print(vibrant_colors)

canvas = np.ones((500, 500, 3)).astype(np.uint8)*255

circle_center = (250, 250)
vibrant_radius = 150
ambient_radius = 190
angle_per_swatch = 360/24



for i, trace in enumerate(main_colors):
    # swatch = trace.paint_trace()
    trace_ambient_color = to_cv2_color(main_colors[i])
    trace_vibrant_color = to_cv2_color(vibrant_colors[i])
    if trace_ambient_color[0] == 255 and trace_ambient_color[1] == 255 and trace_ambient_color[2] == 255:
        continue

    start_angle = hours[i] * angle_per_swatch - 90
    canvas = cv2.ellipse(canvas,circle_center,(ambient_radius, ambient_radius), start_angle, 0, angle_per_swatch,
                            trace_ambient_color, -1)
    canvas = cv2.ellipse(canvas,circle_center,(vibrant_radius, vibrant_radius), start_angle, 0, angle_per_swatch,
                            trace_vibrant_color, -1)

largest_color = main_colors[-1]
vibrant_color = vibrant_colors[-1]
current_ambient_color = to_cv2_color(largest_color)
current_vibrant_color =to_cv2_color(vibrant_color)

vr_percent = 500/(1000+500)

canvas = cv2.circle(canvas, circle_center, vibrant_radius-40, current_ambient_color, -1)
canvas = cv2.circle(canvas, circle_center, int((vibrant_radius-40)*vr_percent**(0.5)), current_vibrant_color, -1)

cv2.imwrite("C:/work/slow_lamp/render_test/render.png", canvas)