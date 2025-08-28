import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
import datetime
from datetime import date, timedelta

record = "C:/work/slow_lamp/june_july_days.txt"
storage_file = open("C:/work/slow_lamp/render_test/simulate_year.txt", "w") 

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

class Trace:
    def __init__(self, main_color, day, hour, traces_storing_mode="single", supplemental_colors=None):
        self.main_color = np.array(main_color[:3])
        self.traces_storing_mode = traces_storing_mode
        self.supplemental_colors = []
        if traces_storing_mode != "single":
            self.supplemental_colors = np.array(supplemental_colors)[:, :3]

        start_date = datetime.datetime(2025, 1, 1, 0, 0, 0 )
        dt = start_date + timedelta(days=day, hours=hour)
        self.timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")

    def print_trace(self):
        main_str = f'mode: {self.traces_storing_mode}; main_color (rgb): ({int(self.main_color[0])}, {int(self.main_color[1])}, {int(self.main_color[2])});'
        supp_str = ''
        if self.traces_storing_mode != "single":
            supp_str = "supplemental_colors: ["
            for color in self.supplemental_colors:
                supp_str += f'({int(color[0])}, {int(color[1])}, {int(color[2])}),'
            supp_str+="];"
        return f'{main_str} {supp_str} @ {self.timestamp} \n'

sunrise_vibrant_colors = []
day_vibrant_colors = []
sunset_vibrant_colors = []
night_vibrant_colors = []

sunrise_ambient_colors = []
day_ambient_colors = []
sunset_ambient_colors = []
night_ambient_colors = []

with open(record, 'r') as file:
    for line in file:
        # Extract main color using regex
        main_match = re.search(r"main_color \(rgb\): \((\d+), (\d+), (\d+)\)", line)
        if main_match:
            main_rgb = tuple(map(int, main_match.groups()))
            if main_rgb[0] == 255 and main_rgb[1] == 255 and main_rgb[2] == 255:
                continue
        # Extract supplemental colors using regex
        supp_match = re.search(r"supplemental_colors: \[(.*?)\]", line)
        if supp_match:
            color_list = supp_match.group(1)
            # Extract all RGB tuples from the list
            all_supp_colors = re.findall(r"\((\d+), (\d+), (\d+)\)", color_list)
            if len(all_supp_colors) >= 2:
                second_rgb = tuple(map(int, all_supp_colors[1]))

        time_match = re.search(r"@ \d{4}-\d{2}-(\d{2}) (\d{2}):\d{2}:\d{2}", line)
        if time_match:
            hour = int(time_match.group(2))
            if hour >= 4 and hour < 6:
                sunrise_ambient_colors.append(main_rgb)
                sunrise_vibrant_colors.append(second_rgb)
            elif hour >= 6 and hour < 20: 
                day_ambient_colors.append(main_rgb)
                day_vibrant_colors.append(second_rgb)
            elif hour >= 20 and hour < 22: 
                sunset_ambient_colors.append(main_rgb)
                sunset_vibrant_colors.append(second_rgb)
            else:
                night_ambient_colors.append(main_rgb)
                night_vibrant_colors.append(second_rgb)

print(day_vibrant_colors)

start_day = 0
end_day = 364

anchors = [0, 60, 70, 120, 180, 240, 300, 360]
sunrise = [7, 7, 6, 6, 4, 6, 6, 7]
sunrise_hours = np.interp(np.arange(0, 365), anchors, sunrise)
dark_hours = np.random.randint(-1, 2, size=(365,))+22 

img_size = 800

canvas = np.ones((img_size, img_size, 3)).astype(np.uint8) *255

circle_center = (img_size // 2, img_size // 2)
# vibrant_radius = 170
ambient_radius = 370
angle_per_swatch = 360/24

center_x, center_y = img_size // 2, img_size // 2
a = 20  # Controls initial radius

max_angle = (end_day- start_day) * 2 * np.pi  
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

for i in range(start_day, end_day):
    # swatch = trace.paint_trace()
    sunrise_hour = sunrise_hours[i]
    dark_hour = dark_hours[i]
    for hour in range(24):
        if abs(hour - sunrise_hour)<=1:
            color_choice = np.random.randint(0, len(sunrise_ambient_colors))
            am_color = sunrise_ambient_colors[color_choice]
            vr_color = sunrise_vibrant_colors[color_choice]
        elif hour > sunrise_hour + 1 and hour < dark_hour - 1: 
            color_choice = np.random.randint(0, len(day_ambient_colors))
            am_color = day_ambient_colors[color_choice]
            vr_color = day_vibrant_colors[color_choice]
        elif abs(hour - dark_hour)<=1: 
            color_choice = np.random.randint(0, len(sunset_ambient_colors))
            am_color = sunset_ambient_colors[color_choice]
            vr_color = sunset_vibrant_colors[color_choice]
        else:
            color_choice = np.random.randint(0, len(night_ambient_colors))
            am_color = night_ambient_colors[color_choice]
            vr_color = night_vibrant_colors[color_choice]

        trace = Trace(am_color, i, hour, traces_storing_mode="levc", supplemental_colors=[vr_color, vr_color, vr_color])
        storage_file.writelines([trace.print_trace()])

        trace_ambient_color = to_cv2_color(am_color)
        trace_vibrant_color = to_cv2_color(vr_color)
        
        start_angle = ((end_day-i)*360) + hour * angle_per_swatch - 90
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
    
    # glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 0] += trace_vibrant_color[0]*gaussian_dot
    # glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 1] += trace_vibrant_color[1]*gaussian_dot
    # glowing_canvas[x-dot_size//2: x + dot_size//2, y-dot_size//2:y+dot_size//2, 2] += trace_vibrant_color[2]*gaussian_dot

largest_color = sunset_ambient_colors[-1]
vibrant_color = sunrise_vibrant_colors[-1]
current_ambient_color = to_cv2_color(largest_color)
current_vibrant_color =to_cv2_color(vibrant_color)

vr_percent = 300/(1000+500)

canvas = cv2.circle(canvas, circle_center, a, current_ambient_color, -1)
canvas = cv2.circle(canvas, circle_center, int((a)*vr_percent**(0.5)), current_vibrant_color, -1)

cv2.imwrite("C:/work/slow_lamp/render_test/render_spiral_year.png", canvas)

storage_file.flush()
storage_file.close()
# glowing_canvas = (glowing_canvas/np.max(glowing_canvas)*255).astype(np.uint8)
# cv2.imwrite("C:/work/slow_lamp/render_test/render_glowing_spiral.png", glowing_canvas)