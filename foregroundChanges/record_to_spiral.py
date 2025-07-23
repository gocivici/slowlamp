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
days = []

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

        time_match = re.search(r"@ \d{4}-\d{2}-(\d{2}) (\d{2}):\d{2}:\d{2}", line)
        if time_match:
            day = int(time_match.group(1))
            if day == 14:
                days.append(15)
            elif day == 13:
                days.append(14)
            elif day == 8:
                days.append(13)
            elif day == 7:
                days.append(12)
            else:
                days.append(day)
            hour = int(time_match.group(2))
            hours.append(hour)


print(main_colors)
print(vibrant_colors)
print(days)

data_length = len(days)
start_day = days[0]

img_size = 500

canvas = np.ones((img_size, img_size, 3)).astype(np.uint8) *255

circle_center = (img_size // 2, img_size // 2)
vibrant_radius = 170
ambient_radius = 210
angle_per_swatch = 360/24

center_x, center_y = img_size // 2, img_size // 2
a = 70  # Controls initial radius

max_angle = (days[-1]- start_day) * 2 * np.pi  
print("max_angle", max_angle/np.pi*180)

b = (ambient_radius - a) / max_angle   # Controls how fast the radius increases
thickness = b * 2 * np.pi
print("thickness", thickness)

dot_size = thickness*3
# gaussian_dot = np.ones((dot_size, dot_size, 3)).astype(np.uint8)
sigma_sq = (dot_size/2.5)**2 # dot_size**2/(np.log(255))/2
gx = np.linspace(-dot_size, dot_size, int(dot_size))
gy = np.linspace(-dot_size, dot_size, int(dot_size))
gxv, gyv = np.meshgrid(gx, gy)
gaussian_dot = np.exp(-(gxv**2+gyv**2)/(2*sigma_sq))
print(np.max(gaussian_dot), np.min(gaussian_dot))
dot_size = int(dot_size)

glowing_canvas = np.zeros((img_size, img_size, 3))

for i in range(1, len(main_colors)+1):
    # swatch = trace.paint_trace()
    trace_ambient_color = to_cv2_color(main_colors[data_length-i])
    trace_vibrant_color = to_cv2_color(vibrant_colors[data_length-i])
    if trace_ambient_color[0] == 255 and trace_ambient_color[1] == 255 and trace_ambient_color[2] == 255:
        # continue
        trace_ambient_color = (0, 0, 0)
        trace_vibrant_color = (0, 0, 0)

    
    start_angle = ((days[data_length-i]-start_day)*360) + hours[data_length-i] * angle_per_swatch - 90
    # print(start_angle)
    radius = int(a + b * ((start_angle + angle_per_swatch/2)/180*np.pi))
    x = int(center_x + radius * np.cos(start_angle + angle_per_swatch/2))
    y = int(center_y + radius * np.sin(start_angle + angle_per_swatch/2))
    # points.append((x, y))
    # print(radius)

    canvas = cv2.ellipse(canvas, circle_center,(radius, radius), start_angle, 0, angle_per_swatch,
                            trace_ambient_color, -1)
    canvas = cv2.ellipse(canvas, circle_center,(int(radius-thickness*0.5), int(radius-thickness*0.5)), start_angle, 0, angle_per_swatch,
                            trace_vibrant_color, -1)
    
    glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 0] += trace_vibrant_color[0]*gaussian_dot
    glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 1] += trace_vibrant_color[1]*gaussian_dot
    glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 2] += trace_vibrant_color[2]*gaussian_dot

largest_color = main_colors[-1]
vibrant_color = vibrant_colors[-1]
current_ambient_color = to_cv2_color(largest_color)
current_vibrant_color =to_cv2_color(vibrant_color)

vr_percent = 300/(1000+500)

canvas = cv2.circle(canvas, circle_center, a, current_ambient_color, -1)
canvas = cv2.circle(canvas, circle_center, int((a)*vr_percent**(0.5)), current_vibrant_color, -1)

cv2.imwrite("C:/work/slow_lamp/render_test/render_spiral.png", canvas)

glowing_canvas = (glowing_canvas/np.max(glowing_canvas)*255).astype(np.uint8)
cv2.imwrite("C:/work/slow_lamp/render_test/render_glowing_spiral.png", glowing_canvas)