import csv
import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
# import datetime
from datetime import datetime, timedelta
from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment


record = "C:/work/slow_lamp/light_simulation/20250723_two_days.txt"
csv_file = open("C:/work/slow_lamp/light_simulation/animation_plan_11x11.csv", "w", newline='') 
writer = csv.writer(csv_file)

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)

# main_pos = [(2, 2), (1, 2), (2, 3), (3, 2), (2, 1)]
# # row major order
# available_pos = [(0, 0), (0, 4), (4, 4), (4, 0)]
# available_pos_dict = {
#     0: [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)],#, (2, 0)], #(0, 3), (1, 2), (2, 1), (3, 0)],
#     1: [(0, 4), (1, 4), (0, 3), (1, 3), (2, 4)],#, (0, 2)], #(3, 4), (2, 3), (1, 2), (0, 1)],
#     2: [(4, 4), (4, 3), (3, 4), (3, 3), (4, 2)],#, (2, 4)], # (4, 1), (3, 2), (2, 3), (1, 4)],
#     3: [(4, 0), (3, 0), (4, 1), (3, 1), (2, 0)]#, (4, 2)], # (1, 0), (2, 1), (3, 2), (4, 3)]
# }
grid_size = 11
vibrant_center = (grid_size//2, grid_size//2)
available_pos = [(0, 0), (0, grid_size-1), (grid_size-1, grid_size-1), (grid_size-1, 0)]
# initialize position array
# Function to check if a point (x, y) is inside the circle
main_pos = []
available_pos_dict = {
    0: [],#, (2, 0)], #(0, 3), (1, 2), (2, 1), (3, 0)],
    1: [],#, (0, 2)], #(3, 4), (2, 3), (1, 2), (0, 1)],
    2: [],#, (2, 4)], # (4, 1), (3, 2), (2, 3), (1, 4)],
    3: []#, (4, 2)], # (1, 0), (2, 1), (3, 2), (4, 3)]
}
def is_point_in_circle(x, y, center, radius):
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

for r in range(grid_size//2):
    current_main_pos = set(main_pos)
    current_available_pos_dict = {
        0: set(available_pos_dict[0]),#, (2, 0)], #(0, 3), (1, 2), (2, 1), (3, 0)],
        1: set(available_pos_dict[1]),#, (0, 2)], #(3, 4), (2, 3), (1, 2), (0, 1)],
        2: set(available_pos_dict[2]),#, (2, 4)], # (4, 1), (3, 2), (2, 3), (1, 4)],
        3: set(available_pos_dict[3])#, (4, 2)], # (1, 0), (2, 1), (3, 2), (4, 3)]
    }
    for x in range(grid_size):
        for y in range(grid_size):
            if r < np.sqrt(grid_size**2/5/np.pi) + 0.3:
                if is_point_in_circle(x, y, vibrant_center, r):
                    if (x, y) not in current_main_pos:
                        main_pos.append((x, y))
                else:
                    for ic, center in enumerate(available_pos):
                        if is_point_in_circle(x, y, center, r):
                            if (x, y) not in current_available_pos_dict[ic]:
                                available_pos_dict[ic].append((x, y))
                            break
            else:
                for ic, center in enumerate(available_pos):
                    if is_point_in_circle(x, y, center, r):
                        if (x, y) not in current_available_pos_dict[ic]:
                            available_pos_dict[ic].append((x, y))
                        break
print(main_pos)
print(available_pos_dict)


base_watt = 150
num_per_group = len(available_pos_dict[0])
max_area_ratio = 1/num_per_group

header = ["frame"]
for i in range(grid_size):
    for j in range(grid_size):
        header.extend([f"light_{i}_{j}_Wt", f"light_{i}_{j}_R", f"light_{i}_{j}_G", f"light_{i}_{j}_B" ])

writer.writerows([header])

class ColorTrack:
    def __init__(self, color, count, position_group = 0):
        self.color = np.array(color)
        # self.age = 0
        # self.missed = 0
        self.count = count
        self.position_group = position_group
        self.position = available_pos[position_group]
        self.watt = base_watt # /len(available_pos_dict[0])

    def update_color(self, new_color_w_count, alpha=0.8):
        self.color = (1 - alpha) * self.color + alpha * np.array(new_color_w_count[1:])
        # self.missed = 0
        self.count += new_color_w_count[0]
        self.count -= min(300, self.count-10)
    
    def update_position(self, new_pos):
        self.position = new_pos

    def print_swatch(self):
        return self.color, self.count, self.watt

start_hour = None
last_hour = None
tracks = []


with open(record, 'r') as file:
    for i, line in enumerate(file):
        # mode: vaooo; main_color (rgb) #18779: (102, 128, 172); supplemental_colors: [ #51700 (21, 22, 20), #29686 (40, 41, 42), #15506 (71, 85, 115), #5918 (189, 213, 228),]; @ 2025-07-23 16:42:07 
        main_match = re.search(r"main_color \(rgb\) #(\d+): \((\d+), (\d+), (\d+)\)", line)
        if main_match:
            main_count = int(main_match.group(1))
            main_rgb = (int(main_match.group(2)), int(main_match.group(3)), int(main_match.group(4)))
            if main_rgb[0] == 255 and main_rgb[1] == 255 and main_rgb[2] == 255:
                continue
        # Extract all colors using regex
        matches = re.findall(r"#(\d+)\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
        if matches:
            colors_w_count = [ (int(count), int(r), int(g), int(b)) for count, r, g, b in matches ]
            print(colors_w_count)
            supp_colors_w_count = colors_w_count

        # time_match = re.search(r"@ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        # if time_match:
        #     time_str = time_match.group(1)
        #     dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        #     print("at hour", time_str)
        #     if start_hour is None:
        #         start_hour = dt
        #     hour = (dt-start_hour).total_seconds()//(60*60)
        hour = i
        # frame,light_0_0_Wt,light_0_0_R,light_0_0_G,light_0_0_B,light_0_1_Wt,light_0_1_R,light_0_1_G,light_0_1_B,...
        # 1,1000,1.0,0.0,0.0,700,0.0,0.0,1.0,...


        if len(tracks) == 0:
            for j, color in enumerate(supp_colors_w_count):
                tracks.append(ColorTrack(color[1:], color[0], position_group=j))   


        n, m = len(tracks), len(supp_colors_w_count)
        cost_matrix = np.full((n, m), np.inf)
        
        for i, track in enumerate(tracks):
            for j, color in enumerate(supp_colors_w_count):
                track_lab_color = rgb_to_lab(track.color)
                new_lab_color = rgb_to_lab(color[1:])
                dist = np.linalg.norm(np.array(track_lab_color)-np.array(new_lab_color))
                # print(dist)
                cost_matrix[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update matched tracks (after linear sum assignment)
        for i, j in zip(row_ind, col_ind):
            tracks[i].update_color(supp_colors_w_count[j])
            
        
        counts = np.array([track.count for track in tracks])
        # print(counts)
        total_count = np.sum(counts)
        counts_percentage = counts/total_count
        max_count_percentage = np.max(counts_percentage)
        light_counts = np.clip((counts_percentage/max_count_percentage*num_per_group), 1, num_per_group).astype(np.int32)
        total_watts = base_watt*light_counts/num_per_group
        watt_percentage = total_watts/np.sum(total_watts)
        track_watts = base_watt/light_counts*(counts_percentage)/watt_percentage
        for i, track in enumerate(tracks):
            track.update_position(available_pos_dict[track.position_group][:light_counts[i]])
            # track.watt = int(base_watt/light_counts[i]*(1+(watt_percentage[i]-counts_percentage[i])/counts_percentage[i]))
            track.watt = int(track_watts[i])
            print(track.position, "watt", track.watt)

        row_data = np.zeros((len(header)))
        # print(row_data)
        for i, track in enumerate(tracks):
            row_data[0] = hour # frame
            for pos in track.position:
                i, j = pos
                light = f"light_{i}_{j}_"
                # print(header.index(light+"Wt"))
                row_data[header.index(light+"Wt")] = track.watt
                row_data[header.index(light+"R")] = track.color[0]/255
                row_data[header.index(light+"G")] = track.color[1]/255
                row_data[header.index(light+"B")] = track.color[2]/255
        
        main_watt = np.sum(track_watts)*main_count/np.sum(np.array([color[0] for color in supp_colors_w_count]))
        print(main_watt)
        
        for pos in main_pos:
            i, j = pos
            light = f"light_{i}_{j}_"
            row_data[header.index(light+"Wt")] = main_watt
            row_data[header.index(light+"R")] = main_rgb[0]/255
            row_data[header.index(light+"G")] = main_rgb[1]/255
            row_data[header.index(light+"B")] = main_rgb[2]/255

        writer.writerows([row_data])
