from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist

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