import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
import cover_sim

cover_img = "archive-sim-correct1.png"
data = cover_sim.retrieve(cover_img)

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

main_colors = []
vibrant_colors = []
hours = []

for record in data:
    # Extract main color using regex
    main_colors.append(record["ac1"])
    vibrant_colors.append(record["vc"])
    hours.append(record["timestamp"])

print(main_colors)
print(vibrant_colors)
# print(days)

# glowing_canvas = np.zeros((img_size, img_size, 3))

for end_i in range(5, len(hours), 15): #len(days)-1):
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

            # swatch = trace.paint_trace()
        trace_ambient_color = to_cv2_color(main_colors[end_i-j])
        trace_vibrant_color = to_cv2_color(vibrant_colors[end_i-j])
        
        start_angle = ((end_i-(j-start_i))//24*360) + hours[end_i-j] * angle_per_swatch - 90
        # print(start_angle)
        radius = int(a + b * ((start_angle + angle_per_swatch/2 + 90)/180*np.pi))
        x = int(center_x + radius * np.cos(start_angle + angle_per_swatch/2))
        y = int(center_y + radius * np.sin(start_angle + angle_per_swatch/2))
        # points.append((x, y))
        # print(radius)

        canvas = cv2.ellipse(canvas, circle_center,(radius, radius), start_angle, 0, angle_per_swatch,
                                trace_ambient_color, -1)
        if radius < thickness:
            canvas = cv2.ellipse(canvas, circle_center,(int(radius*0.5), int(radius*0.5)), start_angle, 0, angle_per_swatch,
                                trace_vibrant_color, -1)
        else:
            canvas = cv2.ellipse(canvas, circle_center,(int(radius-thickness*0.5), int(radius-thickness*0.5)), start_angle, 0, angle_per_swatch,
                                trace_vibrant_color, -1)

    largest_color = main_colors[data_length]
    vibrant_color = vibrant_colors[data_length]
    current_ambient_color = to_cv2_color(largest_color)
    current_vibrant_color =to_cv2_color(vibrant_color)

    vr_percent = 300/(1000+500)

    canvas = cv2.circle(canvas, circle_center, a, current_ambient_color, -1)
    canvas = cv2.circle(canvas, circle_center, int((a)*vr_percent**(0.5)), current_vibrant_color, -1)

    cv2.imwrite(f"C:/work/slow_lamp/render_test/real_data_animation/render_spiral_{end_i:05d}.png", canvas)