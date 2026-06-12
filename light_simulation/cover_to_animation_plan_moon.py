import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist
import random
import cover_sim
import pandas as pd

cover_img = "archive-sim-correct1.png"
cover_start_time = datetime.strptime("2025-11-11 00:53:34", "%Y-%m-%d %H:%M:%S") 
data = cover_sim.retrieve(cover_img)


precise_moon_phase = False
# 2. Define the target time and get the target illumination
current_time = pd.to_datetime('2026-10-08 17:00:00')


record_colors = []
record_counts = []
hours = []
for record in data:
    # Extract main color using regex
    record_colors.append([record["vc"], record["ac1"], record["ac2"], record["ac3"], record["ac4"]])
    record_counts.append([record["vc_px"], record["ac1_px"], record["ac2_px"], record["ac3_px"], record["ac4_px"]])
    # vibrant_colors.append(record["vc"])
    hours.append(record["timestamp"])


record = "C:/work/slow_lamp/light_simulation/20250723_two_days.txt"
csv_file = open("C:/work/slow_lamp/light_simulation/animation_plan_2026-10-08_5pm_hour.csv", "w", newline='') 
csv_coordinates = open("C:/work/slow_lamp/light_simulation/coordinates_freeform_2026-10-08_5pm_hour.csv", "w", newline='') 
writer = csv.writer(csv_file)
coordinate_writer = csv.writer(csv_coordinates)

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

if shape == "square":
    grid_size = 11
    grid_points_x = np.linspace(-5.5, 5.5, grid_size)
    grid_points_y = np.linspace(-5.5, 5.5, grid_size)
    led_points = np.stack(np.meshgrid(grid_points_x, grid_points_y), axis = 2).reshape(-1, 2)
    print(led_points.shape)
    
elif shape == "circle":
    num_led_r = 5 #half
    circle_radius = 11/2
    gap = circle_radius/num_led_r
    led_points = []
    for i in range(num_led_r):
        if i == 0:
            led_points.append([0, 0])
        else:
            r = circle_radius/(num_led_r-1)*i
            arc_theta = gap/r
            for t in np.arange(0, np.pi*2, arc_theta):
                led_points.append([0+r*np.cos(t), 0+r*np.sin(t)])

led_points = np.array(led_points)
# plt.scatter(led_points[:, 0], led_points[:, 1])
# plt.show()
# exit()

num_leds = len(led_points)
base_watt = 70
num_per_group = num_leds//5

c_header = []
for i in range(num_leds):
    c_header.extend([f"light_{i}_x", f"light_{i}_y", f"light_{i}_z" ])
coordinate_writer.writerows([c_header])

row = []
for i in range(num_leds):
    x, y, = led_points[i]
    z = 2
    row.extend([x, y, z])
coordinate_writer.writerows([row])

header = ["frame"]
for i in range(num_leds):
    header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B" ])

writer.writerows([header])

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
    print("all_centers", all_centers)
    print("target_counts", target_counts)
    
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

from load_moon_hourly_phase import read_navy_moon_table
df_2026 = read_navy_moon_table("moon_table_2026.csv", 2026)
df_2025 = read_navy_moon_table("moon_table_2025.csv", 2025)
# 1. Concatenate the dataframes
# Pass them as a list. ignore_index=True creates a fresh, clean integer index 0, 1, 2...
df_combined = pd.concat([df_2025, df_2026], ignore_index=True)

# 2. Sort chronologically 
# (Strictly speaking, if 2025 is first in the list, it's already sorted, 
# but this is a great safety habit before interpolating time series data)
df_combined = df_combined.sort_values(by='Date').reset_index(drop=True)

# Now you can proceed with the hourly resampling on df_combined!
df_hourly = df_combined.set_index('Date').resample('h').interpolate(method='time')

df_hourly.to_csv("hourly_moon_table_25-26.csv")

# 1. Calculate the hour-to-hour change to determine Waxing vs Waning
# fillna(0) handles the very first row which won't have a previous hour to compare to
df_hourly['Illum_Change'] = df_hourly['Illumination'].diff().fillna(0)
                                            

from scipy.signal import find_peaks

# Extract the exact illumination at this hour
target_illum = df_hourly.loc[current_time, 'Illumination']
target_change = df_hourly.loc[current_time, 'Illum_Change']

# Determine the phase direction: True if getting brighter, False if getting darker
is_waxing = target_change > 0 

# 5. Map these times to your specific hourly index starting in Nov 2025
# Replace this with your exact starting datetime
index_start_time = pd.to_datetime(cover_start_time)
print("cover start time", index_start_time)
# 3. Filter for past dates only
df_valid = df_hourly[df_hourly.index <= current_time].copy()
df_search = df_valid[df_valid.index > index_start_time].copy()

# Calculate the baseline absolute difference
df_search['Abs_Diff'] = (df_search['Illumination'] - target_illum).abs()

# 4. The Magic Trick: Penalize the wrong phase
# If we want a waning moon, we artificially inflate the Abs_Diff of all waxing moons 
# by a huge number (e.g., 100) so find_peaks will never select them as a minimum.
if is_waxing:
    wrong_phase_mask = df_search['Illum_Change'] <= 0
else:
    wrong_phase_mask = df_search['Illum_Change'] > 0

df_search.loc[wrong_phase_mask, 'Abs_Diff'] += 100

# 5. Find the peaks (valleys)
# Because we eliminated the opposite phase (which was ~14 days away), the next 
# valid match is guaranteed to be a full lunar cycle (~29.5 days) away.
# A distance of 20 days (20 * 24 hours) is now perfectly safe and won't skip February.
peaks, _ = find_peaks(-df_search['Abs_Diff'], distance=20*24)

# Extract the valid matches
matched_data = df_search.iloc[peaks].copy()

if precise_moon_phase:

    # 6. Map to your specific hourly index
    index_start_time = pd.to_datetime(cover_start_time)
    matched_data['Hour_Index'] = ((matched_data.index - index_start_time).total_seconds() / 3600).astype(int)

    # --- Verification ---
    print(f"Target Date: {current_time}")
    print(f"Target Illumination: {target_illum:.4f}\n")

    print("Matched Past Points (One per cycle):")
    print(matched_data[['Illumination', 'Abs_Diff', 'Hour_Index']])

    # The final arrays you need:
    matched_indices_array = matched_data['Hour_Index'].values
    # matched_illuminations_array = matched_data['Illumination'].values  

else: #same hour of that day 
    current_hour = current_time.hour
    matched_data['Date_norm'] = matched_data.index.normalize() + pd.to_timedelta(current_hour, unit='h')

    matched_data['Hour_Index'] = ((matched_data["Date_norm"] - index_start_time).dt.total_seconds() / 3600).astype(int)

    # print(matched_data)
    
    print(f"Target Date: {current_time}")
    print(f"Target Illumination: {target_illum:.4f}\n")
    print("Matched Past Points (One per cycle):")
    print(matched_data[['Date_norm', 'Illumination', 'Abs_Diff', 'Hour_Index']])

    matched_indices_array = matched_data['Hour_Index'].values

start_hour = None
last_hour = None
tracks = []

for hour in matched_indices_array:
        # time_match = re.search(r"@ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        # if time_match:
        #     time_str = time_match.group(1)
        #     dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        #     print("at hour", time_str)
        #     if start_hour is None:
        #         start_hour = dt
        #     hour = (dt-start_hour).total_seconds()//(60*60)
        main_color_set = record_colors[hour]
        main_color_counts = record_counts[hour]

        supp_colors_w_count = [[main_color_counts[i], *main_color_set[i]] for i in range(1, 5)]
        main_count = main_color_counts[0]
        main_rgb = main_color_set[0]
        
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
        print("main_count", main_count)
        print("counts", counts)
        total_count = np.sum(counts)
        counts_percentage = counts/total_count
        max_count_percentage = np.max(counts_percentage)
        light_counts = np.clip((counts_percentage/max_count_percentage*num_per_group), 1, num_per_group).astype(np.int32)
        print(light_counts)
        # vibrant_count = main_count/np.sum(np.array([color[0] for color in supp_colors_w_count]))*np.sum(light_counts)
        # print(vibrant_count, light_counts)

        assignments, centers = refine_clusters_iteratively(led_points, light_counts,  other_centroids)
        # Print results
        print("Cluster assignments:", assignments)
        print("Cluster centers:", centers)
        
        # Print actual counts per cluster
        all_light_counts = []
        for i in range(5):
            count = assignments.count(i)
            all_light_counts.append(count)
            cluster_name = "vibrant" if i == 0 else f"color_{i}"
            print(f"Cluster {i} ({cluster_name}): {count} LEDs")
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
            print(track.position, "watt", track.watt)

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

        writer.writerows([row_data])
