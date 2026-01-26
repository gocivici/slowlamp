import csv
import numpy as np
import matplotlib.pyplot as plt
import re
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random
import colour 
import correct_color_HD108


record = "/home/slowlamp2/Documents/slowlamp/HD108/20250723_two_days.txt"
# csv_file = open("C:/work/slow_lamp/light_simulation/animation_HD108_freeform.csv", "w", newline='') 
# csv_coordinates = open("C:/work/slow_lamp/light_simulation/coordinates_HD108_freeform.csv", "w", newline='') 
# writer = csv.writer(csv_file)
# coordinate_writer = csv.writer(csv_coordinates)

key_frames = []

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
shape = "circle" # circle
random_centroid = True
plot_cluster = False

# if shape == "square":
#     grid_size = 11
#     grid_points_x = np.linspace(-5.5, 5.5, grid_size)
#     grid_points_y = np.linspace(-5.5, 5.5, grid_size)
#     led_points = np.stack(np.meshgrid(grid_points_x, grid_points_y), axis = 2).reshape(-1, 2)
#     print(led_points.shape)
    
# elif shape == "circle":
#     num_led_r = 5 #half
#     circle_radius = 11/2
#     gap = circle_radius/num_led_r
#     led_points = []
#     for i in range(num_led_r):
#         if i == 0:
#             led_points.append([0, 0])
#         else:
#             r = circle_radius/(num_led_r-1)*i
#             arc_theta = gap/r
#             for t in np.arange(0, np.pi*2, arc_theta):
#                 led_points.append([0+r*np.cos(t), 0+r*np.sin(t)])

# led_points = np.array(led_points)
# plt.scatter(led_points[:, 0], led_points[:, 1])
# plt.show()
# exit()

nums_per_row = np.array([38, 38, 37, 36, 35, 34, 32, 29, 26, 21, 13])
nums_per_row = nums_per_row[::-1]
led_points = np.zeros((np.sum(nums_per_row), 2))

# Horizontal spacing (pitch) and Vertical spacing between rows
pitch = 7 
row_spacing = 12

current_index = 0

for i, num in enumerate(nums_per_row):
    # Calculate indices for the current row
    end_index = current_index + num
    
    # Calculate Y (vertical position)
    led_points[current_index:end_index, 1] = i * row_spacing
    
    # Calculate X (centered at 0)
    # Total width of this row is (num - 1) * pitch
    row_width = (num - 1) * pitch
    x = np.linspace(-row_width/2, row_width/2, num)
    
    # Handle Zig-Zag (Reverse every other row)
    if i % 2 == 1:
        led_points[current_index:end_index, 0] = x[::-1]
    else:
        led_points[current_index:end_index, 0] = x
        
    current_index = end_index

print(led_points)

# plt.figure()
# colors = np.ones((len(led_points), 3))
# colors[:, 1] = np.linspace(0, 1, len(led_points))
# plt.scatter(led_points[:, 0], led_points[:, 1], c = colors)
# plt.show()

num_leds = len(led_points)
base_watt = 70
num_per_group = num_leds//5

# c_header = []
# for i in range(num_leds):
#     c_header.extend([f"light_{i}_x", f"light_{i}_y", f"light_{i}_z" ])
# coordinate_writer.writerows([c_header])

# row = []
# for i in range(num_leds):
#     x, y, = led_points[i]
#     z = 2
#     row.extend([x, y, z])
# coordinate_writer.writerows([row])

header = ["frame"]
for i in range(num_leds):
    header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B" ])

# writer.writerows([header])

class ColorTrack:
    def __init__(self, color, count, position_group = 0):
        self.color = np.array(color)
        # self.age = 0
        # self.missed = 0
        self.count = count
        # self.position_group = position_group
        self.centroid = [0, 0]
        self.position = [0]
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


def distribute_leds_to_clusters(led_coordinates, target_counts, other_centers):
    """
    Distribute LED points into 5 clusters with specified counts.
    
    Parameters:
    - led_coordinates: list of [x, y] coordinates for each LED
    - light_counts: list of 4 integers specifying desired count for each non-vibrant cluster
    - vibrant_count: integer specifying desired count for vibrant (center) cluster
    
    Returns:
    - cluster_assignments: list of cluster IDs (0=vibrant, 1-4=other colors)
    - cluster_centers: list of [x, y] coordinates for cluster centers
    """
    
    led_coords = np.array(led_coordinates)
    num_leds = len(led_coords)
    
    # Calculate center of all LEDs for vibrant cluster
    center_x = np.mean(led_coords[:, 0])
    center_y = np.mean(led_coords[:, 1])
    vibrant_center = np.array([center_x, center_y])
    
    # Generate random centroids around the vibrant center for other clusters
    # Use a reasonable radius based on the spread of LEDs
    spread = max(np.std(led_coords[:, 0]), np.std(led_coords[:, 1]))
    radius = spread * 0.8  # Adjust this factor as needed
    
    # other_centers = []
    # for i in range(4):
    #     angle = (2 * np.pi * i / 4) + random.uniform(-np.pi/8, np.pi/8)  # Add some randomness
    #     offset_x = radius * np.cos(angle)
    #     offset_y = radius * np.sin(angle)
    #     center = np.array([center_x + offset_x, center_y + offset_y])
    #     other_centers.append(center)
    
    # Combine all centers
    all_centers = [vibrant_center] + other_centers
    # target_counts = [vibrant_count] + list(light_counts)
    # print("all_centers", all_centers)
    # print("target_counts", target_counts)
    
    # Initialize cluster assignments
    cluster_assignments = [-1] * num_leds
    assigned_leds = set()
    
    # For each cluster, assign the closest unassigned LEDs
    for cluster_id in range(5):
        target_count = target_counts[cluster_id]
        cluster_center = all_centers[cluster_id]
        
        # Calculate distances from cluster center to all unassigned LEDs
        unassigned_indices = [i for i in range(num_leds) if i not in assigned_leds]
        if not unassigned_indices:
            break
        
        unassigned_coords = led_coords[unassigned_indices]
        distances = cdist(np.array(cluster_center).reshape(1, 2), unassigned_coords)[0]
        
        # Sort by distance and take the closest LEDs up to target count
        sorted_indices = np.argsort(distances)
        num_to_assign = int(min(target_count, len(unassigned_indices)))
        
        for i in range(num_to_assign):
            led_index = unassigned_indices[sorted_indices[i]]
            cluster_assignments[led_index] = cluster_id
            assigned_leds.add(led_index)
    
    # Assign any remaining unassigned LEDs to the closest cluster
    # for i in range(num_leds):
    #     if cluster_assignments[i] == -1:
    #         distances = [np.linalg.norm(led_coords[i] - center) for center in all_centers]
    #         cluster_assignments[i] = np.argmin(distances)
    
    return cluster_assignments, all_centers

def refine_clusters_iteratively(led_coordinates, target_counts, other_centroids, iterations=3):
    """
    Refine cluster assignments through multiple iterations to better match target counts.
    """
    led_coords = np.array(led_coordinates)
    # target_counts = [vibrant_count] + list(light_counts)
    
    # Initial assignment
    cluster_assignments, centers = distribute_leds_to_clusters(led_coordinates, target_counts, other_centroids)
    
    for iteration in range(iterations):
        # Update centers based on current assignments
        new_centers = []
        for cluster_id in range(5):
            cluster_leds = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
            if cluster_leds:
                cluster_coords = led_coords[cluster_leds]
                new_center = np.mean(cluster_coords, axis=0)
                new_centers.append(new_center)
            else:
                new_centers.append(centers[cluster_id])  # Keep old center if no LEDs assigned
        
        centers = new_centers
        
        # Reassign LEDs based on updated centers, respecting target counts
        cluster_assignments = [-1] * len(led_coordinates)
        assigned_leds = set()
        
        for cluster_id in range(5):
            target_count = target_counts[cluster_id]
            cluster_center = centers[cluster_id]
            
            unassigned_indices = [i for i in range(len(led_coordinates)) if i not in assigned_leds]
            if not unassigned_indices:
                break
                
            unassigned_coords = led_coords[unassigned_indices]
            distances = cdist([cluster_center], unassigned_coords)[0]
            
            sorted_indices = np.argsort(distances)
            num_to_assign = int(min(target_count, len(unassigned_indices)))
            
            for i in range(num_to_assign):
                led_index = unassigned_indices[sorted_indices[i]]
                cluster_assignments[led_index] = cluster_id
                assigned_leds.add(led_index)
        
        # Assign remaining LEDs
        # for i in range(len(led_coordinates)):
        #     if cluster_assignments[i] == -1:
        #         distances = [np.linalg.norm(led_coords[i] - center) for center in centers]
        #         cluster_assignments[i] = np.argmin(distances)
    
    return cluster_assignments, centers

def convert_frame_to_oklab(frame):
    oklab_frame = np.zeros((len(header)))
    oklab_frame[0] = frame[0]
    for i in range(1, len(frame), 4):
        index = (i - 1) // 4
        if index >= num_leds:
            continue
        r = float(frame[i + 1])
        g = float(frame[i + 2])
        b = float(frame[i + 3])
        color_rgb = (r, g, b)
        color_oklab = colour.convert(color_rgb, "sRGB", "Oklab")
        oklab_frame[i] = frame[i]
        oklab_frame[i+1] = color_oklab[0]
        oklab_frame[i+2] = color_oklab[1]
        oklab_frame[i+3] = color_oklab[2]
    return oklab_frame

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
                # main_count = None
                print("Skipping........")
                continue
        # Extract all colors using regex
        matches = re.findall(r"#(\d+)\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
        if matches:
            colors_w_count = [ (int(count), int(r), int(g), int(b)) for count, r, g, b in matches ]
            print(colors_w_count)
            supp_colors_w_count = colors_w_count
        else:
            print("No match??")
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
            # Calculate center of all LEDs for vibrant cluster
            center_x = np.mean(led_points[:, 0])
            center_y = np.mean(led_points[:, 1])

            # Generate random centroids around the vibrant center for other clusters
            # Use a reasonable radius based on the spread of LEDs
            spread = max(np.std(led_points[:, 0]), np.std(led_points[:, 1]))
            radius = spread * 0.8  # Adjust this factor as needed
            
            other_centers = []
            for i in range(4):
                angle = (2 * np.pi * i / 4) + random.uniform(-np.pi/8, np.pi/8)  # Add some randomness
                offset_x = radius * np.cos(angle)
                offset_y = radius * np.sin(angle)
                center = np.array([center_x + offset_x, center_y + offset_y])
                other_centers.append(center)

            for j, color in enumerate(supp_colors_w_count):
                track = ColorTrack(color[1:], color[0], position_group=j)
                track.centroid = other_centers[j]
                tracks.append(track)   
                


        n, m = len(tracks), len(supp_colors_w_count)
        cost_matrix = np.full((n, m), np.inf)
        
        other_centroids = []
        for i, track in enumerate(tracks):
            other_centroids.append(led_points[np.random.choice(track.position, 1)])
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
            
        
        counts = np.array([main_count]+[color[0] for color in supp_colors_w_count])
        # print("main_count", main_count)
        # print("counts", counts)
        total_count = np.sum(counts)
        counts_percentage = counts/total_count
        max_count_percentage = np.max(counts_percentage)
        light_counts = np.clip((counts_percentage/max_count_percentage*num_per_group), 1, num_per_group).astype(np.int32)
        # print(light_counts)
        # vibrant_count = main_count/np.sum(np.array([color[0] for color in supp_colors_w_count]))*np.sum(light_counts)
        # print(vibrant_count, light_counts)

        assignments, centers = refine_clusters_iteratively(led_points, light_counts,  other_centroids)
        # Print results
        # print("Cluster assignments:", assignments)
        # print("Cluster centers:", centers)
        
        # Print actual counts per cluster
        all_light_counts = []
        for i in range(5):
            count = assignments.count(i)
            all_light_counts.append(count)
            cluster_name = "vibrant" if i == 0 else f"color_{i}"
            # print(f"Cluster {i} ({cluster_name}): {count} LEDs")
        all_light_counts = np.array(all_light_counts)

        total_watts = base_watt*all_light_counts
        watt_percentage = total_watts/np.sum(total_watts)
        track_watts = base_watt*(counts_percentage)/watt_percentage
        assignments_arr = np.array(assignments)

        if plot_cluster:
            plt.figure()
        for i, track in enumerate(tracks):
            assigned_indices = np.where(assignments_arr == i+1)[0]
            track.update_position(assigned_indices)
            if plot_cluster:
                assigned_positions = led_points[assigned_indices]
                plt.scatter(assigned_positions[:, 0], assigned_positions[:, 1])
            # track.watt = int(base_watt/light_counts[i]*(1+(watt_percentage[i]-counts_percentage[i])/counts_percentage[i]))
            track.watt = int(track_watts[i+1])
            # print(track.position, "watt", track.watt)

        if plot_cluster:
            assigned_indices = np.where(assignments_arr == 0)[0]
            assigned_positions = led_points[assigned_indices]
            plt.scatter(assigned_positions[:, 0], assigned_positions[:, 1], color = "black")
            plt.show()

        row_data = np.zeros((len(header)))
        # print(row_data)
        for i, track in enumerate(tracks):
            row_data[0] = hour # frame
            for pos in track.position:
                light_index = pos
                light = f"light_{light_index}_"
                # print(header.index(light+"Wt"))
                row_data[header.index(light+"Wt")] = track.watt
                row_data[header.index(light+"R")] = track.color[0]/255
                row_data[header.index(light+"G")] = track.color[1]/255
                row_data[header.index(light+"B")] = track.color[2]/255
        
        main_watt = np.sum(track_watts)*main_count/np.sum(np.array([color[0] for color in supp_colors_w_count]))
        print(main_watt)
        
        for pos in np.where(assignments_arr == 0)[0]:
            light_index = pos
            light = f"light_{light_index}_"
            row_data[header.index(light+"Wt")] = main_watt
            row_data[header.index(light+"R")] = main_rgb[0]/255
            row_data[header.index(light+"G")] = main_rgb[1]/255
            row_data[header.index(light+"B")] = main_rgb[2]/255

        # writer.writerows([row_data])

        key_frames.append(convert_frame_to_oklab(row_data))

key_frames = np.array(key_frames)

key_frame_duration = 1 # seconds
fps = 15

# start_color_oklab = colour.convert(start_color_rgb, "sRGB", "Oklab")
last_color = np.zeros((len(header)))

frame_indices = key_frames[:, 0]
animation_length = int(frame_indices[-1]*key_frame_duration*fps)
animation_plan = np.zeros((animation_length, len(header)))
time_arr = np.linspace(0, frame_indices[-1]+1, animation_length) 

def interpolate_non_zeros_stretched(arr_X, arr_Y, new_X):
    non_zero_indices = np.where(arr_Y != 0)[0]
    if len(non_zero_indices) == 0:
        return np.zeros_like(new_X) # zero as black for OkLab? 
    x_known = arr_X[non_zero_indices]
    y_known = arr_Y[non_zero_indices]
    y_interpolated = np.interp(new_X, x_known, y_known)
    return y_interpolated


for i in range(1, len(header), 4):
    print("LED:", (i-1)//4)
    watt = key_frames[:, i]
    l = key_frames[:, i+1]
    a = key_frames[:, i+2]
    b = key_frames[:, i+3]

    l_full = interpolate_non_zeros_stretched(frame_indices, l, time_arr)
    a_full = interpolate_non_zeros_stretched(frame_indices, a, time_arr)
    b_full = interpolate_non_zeros_stretched(frame_indices, b, time_arr)
    oklab_full = np.stack([l_full, a_full, b_full], axis = 1)
    # print("oklab_full.shape", oklab_full.shape)

    lab_full = [] 
    for flab in oklab_full:
        color_lab = colour.convert(flab, "Oklab", "CIE Lab")
        lab_full.append(color_lab)

    lab_full = np.array(lab_full)
    # print(lab_full[0])
    
    input_pixels = correct_color_HD108.correct_color_from_lab(lab_full)
    # print(input_pixels[0])

    input_brightness = np.clip((watt/base_watt)*10, 1, 10)
    animation_brightness = interpolate_non_zeros_stretched(frame_indices, input_brightness, time_arr)

    animation_plan[:, i] = animation_brightness
    animation_plan[:, i+1 : i+4] = input_pixels

print("___________________done generating plan______________", animation_plan.shape)

# exit()

import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)                                            # Open SPI bus 0, device 0
spi.max_speed_hz = 15000000                               # Set speed to 10MHz: 32000000 maximum

def send_hd108_colors_with_brightness(colors_16bit):
    
    data = []
    data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
    
    for brightness, r16, g16, b16 in list(colors_16bit):
        brightness, r16, g16, b16 = map(int, (brightness, r16, g16, b16))
        
        # print(int(r16/255), int(g16/255), int(b16/255))
        if brightness <= 0:
            brightness_frame = (1 << 15) | (1 << 10) | (1 << 5) | 1
        else:
            brightness_frame = (1 << 15) | (brightness << 10) | (brightness << 5) | brightness
        
        data.extend([
            (brightness_frame >> 8) & 0xFF, brightness_frame & 0xFF,             # Brightness
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
        
    num_end = 2 * (len(colors_16bit) + 1) 
    data.extend([0xFF] * num_end)  # End frame
    spi.writebytes(data)

print(animation_plan[0])
print(animation_plan[-10])

for f in animation_plan:
    frame = f[1:].reshape(-1, 4)/3
    send_hd108_colors_with_brightness(frame)
    time.sleep(1/fps)

np.savez("animation_plan.npz", animation_plan=animation_plan)

colors_16bit = [(1, 0, 0, 0)]*num_leds
send_hd108_colors_with_brightness(colors_16bit)
send_hd108_colors_with_brightness(colors_16bit)
spi.close()
