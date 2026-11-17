# Verbatim extract of the animation path from integrated/fgc_arnab.py @ b40801c.
# fgc_arnab.py can't be imported (it opens the camera / SPI / config at import time),
# so this file reproduces only the module globals and functions the animation needs.
# Do not edit the function bodies below; re-extract from fgc_arnab.py instead.
#   globals    : fgc_arnab.py lines 159-169
#   rgb_to_lab : fgc_arnab.py lines 192-195
#   init_tracks .. interpolate_two_frames : fgc_arnab.py lines 591-753

import time
import random

import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment

import helper_classes

led_points = helper_classes.get_led_points_23cm()

num_leds = len(led_points)
base_watt = 70
num_per_group = num_leds//5

header = ["frame"]
for i in range(num_leds):
    header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B" ])

white_frame = [[2, 2**16-1, 2**16-1, 2**16-1]]*num_leds


def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)

def init_tracks(trace, tracks):
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

    for j, (color, count) in enumerate(zip(trace.supplemental_colors, trace.supplemental_counts)):
        track = helper_classes.ColorTrack(color, count, base_watt, position_group=j)
        track.centroid = other_centers[j]
        tracks.append(track) 

def generate_frame_from_trace_tracks(trace, tracks):

    n, m = len(tracks), len(trace.supplemental_colors)
    cost_matrix = np.full((n, m), np.inf)
    
    other_centroids = []
    for i, track in enumerate(tracks):
        other_centroids.append(led_points[np.random.choice(track.position, 1)])
        for j, color in enumerate(trace.supplemental_colors):
            track_lab_color = rgb_to_lab(track.color)
            new_lab_color = rgb_to_lab(color)
            dist = np.linalg.norm(np.array(track_lab_color)-np.array(new_lab_color))
            # print(dist)
            cost_matrix[i, j] = dist
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Update matched tracks (after linear sum assignment)
    for i, j in zip(row_ind, col_ind):
        tracks[i].update_color(trace.supplemental_colors[j], trace.supplemental_counts[j])
        
    
    counts = np.array([trace.main_count]+[c for c in trace.supplemental_counts])
    print("main_count", trace.main_count)
    print("counts", counts)
    total_count = np.sum(counts)
    if total_count == 0:
        row_data = np.zeros((num_leds*4 + 1))
        row_data[0] = int(time.time()) # frame place holder
        return helper_classes.convert_frame_to_oklab(row_data)

    counts_percentage = counts/total_count
    max_count_percentage = np.max(counts_percentage)
    light_counts = np.clip((counts_percentage/max_count_percentage*num_per_group), 1, num_per_group).astype(np.int32)
    print(light_counts)
    # vibrant_count = main_count/np.sum(np.array([color[0] for color in supp_colors_w_count]))*np.sum(light_counts)
    # print(vibrant_count, light_counts)

    assignments, centers = helper_classes.refine_clusters_iteratively(led_points, light_counts,  other_centroids)
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

    for i, track in enumerate(tracks):
        assigned_indices = np.where(assignments_arr == i+1)[0]
        track.update_position(assigned_indices)
        # track.watt = int(base_watt/light_counts[i]*(1+(watt_percentage[i]-counts_percentage[i])/counts_percentage[i]))
        track.watt = int(track_watts[i+1])
        print(track.position, "watt", track.watt)

    row_data = np.zeros((num_leds*4 + 1))
    # print(row_data)
    for i, track in enumerate(tracks):
        row_data[0] = int(time.time()) # frame place holder
        for pos in track.position:
            light_index = pos
            light = f"light_{light_index}_"
            # print(header.index(light+"Wt"))
            row_data[header.index(light+"Wt")] = track.watt
            row_data[header.index(light+"R")] = track.color[0]/255
            row_data[header.index(light+"G")] = track.color[1]/255
            row_data[header.index(light+"B")] = track.color[2]/255
    
    main_watt = np.sum(track_watts)*trace.main_count/np.sum(np.array(trace.supplemental_counts))
    print(main_watt)
    
    for pos in np.where(assignments_arr == 0)[0]:
        light_index = pos
        light = f"light_{light_index}_"
        row_data[header.index(light+"Wt")] = main_watt
        row_data[header.index(light+"R")] = trace.main_color[0]/255
        row_data[header.index(light+"G")] = trace.main_color[1]/255
        row_data[header.index(light+"B")] = trace.main_color[2]/255

    # writer.writerows([row_data])

    return helper_classes.convert_frame_to_oklab(row_data)
    
def interpolate_two_frames(key_frame1, key_frame2, animation_length):
    if animation_length <= 0:
        animation_plan = np.zeros((1, len(key_frame1)))
        return animation_plan

    key_frames = np.stack([key_frame1, key_frame2], axis = 0)

    frame_indices = key_frames[:, 0]
    animation_plan = np.zeros((animation_length, len(key_frame1)))
    time_arr = np.linspace(frame_indices[0], frame_indices[-1]+1, animation_length) 

    for i in range(1, len(key_frame1), 4):
        # print("LED:", (i-1)//4)
        watt = key_frames[:, i]
        l = key_frames[:, i+1]
        a = key_frames[:, i+2]
        b = key_frames[:, i+3]

        l_full = helper_classes.interpolate_stretched(frame_indices, l, time_arr)
        a_full = helper_classes.interpolate_stretched(frame_indices, a, time_arr)
        b_full = helper_classes.interpolate_stretched(frame_indices, b, time_arr)
        oklab_full = np.stack([l_full, a_full, b_full], axis = 1)
        # print("oklab_full.shape", oklab_full.shape)

        # lab_full = [] 
        # for flab in oklab_full:
        #     
        #     color_lab = colour.convert(flab, "Oklab", "CIE Lab")
        #     lab_full.append(color_lab)

        # lab_full = np.array(lab_full)

        rgb_full = helper_classes.oklab_to_srgb_255(oklab_full)
        # print(rgb_full[0])
        
        #TODO: not color correcting yet
        # input_pixels = correct_color_HD108.correct_color_from_lab(lab_full)
        # print(input_pixels[0])
        input_pixels = np.clip(rgb_full*255, 0, 2**16-1).round()
        # print(input_pixels[0])

        input_brightness = np.clip((watt/base_watt)*10, 1, 6)
        animation_brightness = helper_classes.interpolate_stretched(frame_indices, input_brightness, time_arr)

        animation_plan[:, i] = animation_brightness
        animation_plan[:, i+1 : i+4] = input_pixels
    
    return animation_plan
