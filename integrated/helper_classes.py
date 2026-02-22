from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import copy
from sklearn.cluster import KMeans
import colour 

color_swatch_size = 40

def get_led_points_8cm():
    led_points = np.zeros((55, 2))
    # row 7
    y = -3*12 #mm
    x = np.arange(4)*7 - 3.5 - 7
    led_points[0:4, 1] = y
    led_points[0:4, 0] = x[::-1]

    # row 6
    y = -2*12 #mm
    x = np.arange(8)*7 - 3.5 - 7*3
    led_points[4:12, 0] = x
    led_points[4:12, 1] = y

    # row 3
    y = 12 #mm
    x = np.arange(10)*7 - 3.5 - 7*4
    led_points[12:22, 1] = y
    led_points[12:22, 0] = x[::-1]

    # row 4
    y = 0 #mm
    x = np.arange(11)*7 - 7*5
    led_points[22:33, 1] = y
    led_points[22:33, 0] = x

    #row 5 
    y = -12 #mm
    x = np.arange(10)*7 - 3.5 - 7*4
    led_points[33:43, 1] = y
    led_points[33:43, 0] = x[::-1]

    # row 2
    y = 2*12 #mm
    x = np.arange(8)*7 - 3.5 - 7*3
    led_points[43:51, 0] = x
    led_points[43:51, 1] = y

    # row 1
    y = 3*12 #mm
    x = np.arange(4)*7 - 3.5 - 7
    led_points[51:55, 1] = y
    led_points[51:55, 0] = x[::-1]

    # print(led_points)
    return led_points

def get_led_points_10inch():
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

    return led_points

def get_led_points_23cm():
    nums_per_row = np.array([8, 16, 21, 24, 27, 29, 30, 31, 32, 32, 31, 30, 29, 27, 24, 21, 16, 8 ])
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

    return led_points

class ColorTrack:
    def __init__(self, color, count, base_watt, position_group = 0):
        self.color = np.array(color)
        # self.age = 0
        # self.missed = 0
        self.count = count
        # self.position_group = position_group
        self.centroid = [0, 0]
        self.position = [0]
        self.watt = base_watt # /len(available_pos_dict[0])

    def update_color(self, new_color, count, alpha=0.8):
        self.color = (1 - alpha) * self.color + alpha * np.array(new_color)
        # self.missed = 0
        self.count += count
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

def convert_frame_to_oklab(frame):
    oklab_frame = np.zeros_like(frame)
    oklab_frame[0] = frame[0]
    for i in range(1, len(frame), 4):
        index = (i - 1) // 4
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

def interpolate_stretched(arr_X, arr_Y, new_X):
    x_known = arr_X
    y_known = arr_Y
    y_interpolated = np.interp(new_X, x_known, y_known)
    return y_interpolated

class Trace:
    def __init__(self, main_color, count, traces_storing_mode="single", 
                 supplemental_colors=None, supplemental_counts=None):
        self.main_color = np.array(main_color[:3])
        self.main_count = count
        self.traces_storing_mode = traces_storing_mode
        self.supplemental_colors = []
        if traces_storing_mode != "single":
            self.supplemental_colors = np.array(supplemental_colors)[:, :3]
            self.supplemental_counts = supplemental_counts
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def print_trace(self):
        main_str = f'mode: {self.traces_storing_mode}; \
main_color (rgb) #{self.main_count}: ({int(self.main_color[0])}, {int(self.main_color[1])}, {int(self.main_color[2])});'
        supp_str = ''
        if self.traces_storing_mode != "single":
            supp_str = "supplemental_colors: ["
            for color, count in zip(self.supplemental_colors, self.supplemental_counts):
                supp_str += f' #{count} ({int(color[0])}, {int(color[1])}, {int(color[2])}),'
            supp_str+="];"
        return f'{main_str} {supp_str} @ {self.timestamp} \n'
    
    def paint_trace(self):
        global color_swatch_size
        swatch = np.ones((color_swatch_size, color_swatch_size*(1+len(self.supplemental_colors)), 3)).astype(np.uint8)
        swatch[:, :color_swatch_size] = self.main_color[::-1]
        if self.traces_storing_mode != "single":
            for i, color in enumerate(self.supplemental_colors):
                swatch[:, color_swatch_size*(i+1): color_swatch_size*(i+2)] = color[::-1]

        return swatch
    

def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

def extract_colors_kmeans(image, num_colors, resize=True, resize_max=200, color_mode="bgr"):

    if resize:
        image = resize_to_max(image, resize_max)

    if color_mode == "lab":
        # print("lab mode")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    img_np = copy.copy(image)
    pixels = img_np.reshape(-1, 3)

    # Fit KMeans to pixel data
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    labels = kmeans.predict(pixels)

    label_counts = np.bincount(labels)

    # Get dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    if color_mode == "lab":
        rgb_colors = cv2.cvtColor(np.array([colors]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
        return rgb_colors, label_counts
    return colors, label_counts

def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    h1_prime += 360 if h1_prime < 0 else 0
    h2_prime += 360 if h2_prime < 0 else 0
    
    avg_H_prime = (h1_prime + h2_prime + 360) / 2.0 if np.abs(h1_prime - h2_prime) > 180 else (h1_prime + h2_prime) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(avg_H_prime - 30)) + 0.24 * np.cos(np.radians(2 * avg_H_prime)) + 0.32 * np.cos(np.radians(3 * avg_H_prime + 6)) - 0.20 * np.cos(np.radians(4 * avg_H_prime - 63))
    
    delta_h_prime = h2_prime - h1_prime
    delta_h_prime -= 360 if delta_h_prime > 180 else 0
    delta_h_prime += 360 if delta_h_prime < -180 else 0
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2.0)
    
    S_L = 1 + ((0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2))
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    
    R_T = -2 * np.sqrt(avg_C_prime**7 / (avg_C_prime**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-(((avg_H_prime - 275) / 25)**2))))
    
    delta_E = np.sqrt((delta_L_prime / S_L)**2 + (delta_C_prime / S_C)**2 + (delta_H_prime / S_H)**2 + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))
    return delta_E


class FastTrack:
    def __init__(self, color, count):
        self.color = np.array(color)
        self.age = 0
        self.missed = 0
        self.count = count

    def update(self, new_color, new_count, alpha=0.3):
        self.color = (1 - alpha) * self.color + alpha * np.array(new_color)
        self.count = (1 - alpha) * self.count + alpha * new_count
        self.missed = 0
        self.age += 1

    def print_swatch(self):
        return self.color, str(self.count), "+"*self.age, "-"*self.missed