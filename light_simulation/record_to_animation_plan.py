import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import re
from datetime import datetime, timedelta
import cover_sim
from zoneinfo import ZoneInfo
import helper_classes
import time

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import random
import csv

save_cover = False
recompute_animation = True
random_centers = False
save_csv = False
plot_stats = True

using_HD108 = False
animation_fps = 4
total_length = 10
base_watt = 5
clip_min = 0
clip_max = 7
clip_ratio = base_watt

first_x = 50
npz_file = "july21-oct9_605-test.npz"
csv_path = "july21-oct9_605-test.csv"


if using_HD108:
    import spidev

    spi = spidev.SpiDev()
    spi.open(0, 0)                                            # Open SPI bus 0, device 0
    spi.max_speed_hz = 15000000                               # Set speed to 10MHz: 32000000 maximum

    def send_hd108_colors_with_brightness(colors_16bit):
        
        data = []
        data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
        
        for led_i, (brightness, r16, g16, b16) in enumerate(list(colors_16bit)):
            brightness, r16, g16, b16 = map(int, (brightness, r16, g16, b16))
            # if led_i > 390:
              #  print("int values for LED", led_i, ": ",  brightness, r16, g16, b16)
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
        spi.writebytes2(data)

def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)


if save_cover:
    record_files = ["20260721_143553_fgc_arnab.data"]

    start_hour = None
    data = {}

    for j, filename in enumerate(record_files):
        record = "./light_simulation/"+filename
        with open(record, 'r') as file:
            for line in file:
                # mode: vaooo; main_color (rgb) #18779: (102, 128, 172); supplemental_colors: [ #51700 (21, 22, 20), #29686 (40, 41, 42), #15506 (71, 85, 115), #5918 (189, 213, 228),]; @ 2025-07-23 16:42:07 
                main_match = re.search(r"main_color \(rgb\) #(\d+): \((\d+), (\d+), (\d+)\)", line)
                if main_match:
                    main_count = int(main_match.group(1))
                    main_rgb = (int(main_match.group(2)), int(main_match.group(3)), int(main_match.group(4)))
                    if main_rgb[0] == 255 and main_rgb[1] == 255 and main_rgb[2] == 255:
                        continue
                    main_rgb_w_count = (int(main_match.group(1)), int(main_match.group(2)), int(main_match.group(3)), int(main_match.group(4)))
                # Extract all colors using regex
                matches = re.findall(r"#(\d+)\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
                if matches:
                    colors_w_count = [ (int(count), int(r), int(g), int(b)) for count, r, g, b in matches ]
                    print(colors_w_count)
                    supp_colors_w_count = colors_w_count

                time_match = re.search(r"@ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if time_match:
                    time_str = time_match.group(1)
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                    if start_hour is None:
                        start_hour = dt

                    current_hour_utc = dt.replace(tzinfo=ZoneInfo("America/Vancouver"))
                    ts = current_hour_utc.timestamp()//3600
                    
                    hour_from_start = (dt-start_hour).total_seconds()//(60*60)
                    record = [main_rgb_w_count[1:], main_rgb_w_count[0],
                            supp_colors_w_count[0][1:], supp_colors_w_count[0][0], supp_colors_w_count[1][1:], supp_colors_w_count[1][0],
                            supp_colors_w_count[2][1:], supp_colors_w_count[2][0], supp_colors_w_count[3][1:], supp_colors_w_count[3][0], ts]
                    data[hour_from_start] = record

                    
                    
    num_hours = (dt-start_hour).total_seconds()//(60*60) 

    for h in range(0, int(num_hours)):
        if h in data:
            record = data[h]
            if save_cover:
                cover_sim.save(*record)
        else:
            current_hour = start_hour + timedelta(hours = h)
            current_hour_utc = current_hour.replace(tzinfo=ZoneInfo("America/Vancouver"))
            ts = current_hour_utc.timestamp()//3600
            record = ((255, 255, 255), 0, (255, 255, 255), 0,
                            (255, 255, 255), 0, (255, 255, 255), 0,
                            (255, 255, 255), 0, ts)
        # cover_sim.save(*record)

    # print("sunrise_ambient_colors", sunrise_ambient_colors)
led_points = helper_classes.get_led_points_pcb()
num_leds = len(led_points)

if recompute_animation:
    cover_img = "archive-sim.png"
    # cover_start_time = datetime.strptime("2025-11-11 00:53:34", "%Y-%m-%d %H:%M:%S") 
    data = cover_sim.retrieve(filename=cover_img)

    existing_record_colors = []
    existing_record_counts = []
    existing_hours = []
    moon_queue = []

    num_per_group = num_leds//5

    header = ["frame"]
    for i in range(num_leds):
        header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B" ])

    for record in data:
        # Extract main color using regex
        if random_centers:
            existing_record_colors.append([np.random.randint(0, 255, 3), np.random.randint(0, 255, 3), np.random.randint(0, 255, 3), np.random.randint(0, 255, 3), np.random.randint(0, 255, 3)])
        else:
            existing_record_colors.append([record["vc"], record["ac1"], record["ac2"], record["ac3"], record["ac4"]])
        existing_record_counts.append([record["vc_px"], record["ac1_px"], record["ac2_px"], record["ac3_px"], record["ac4_px"]])
        # vibrant_colors.append(record["vc"])
        existing_hours.append(record["timestamp"])

    # print("existing hours len", len(existing_hours))
    for matched_i in range(0, first_x):    
        trace = helper_classes.Trace(existing_record_colors[matched_i][0], existing_record_counts[matched_i][0], 
                            traces_storing_mode="vaooo", 
                            supplemental_colors= existing_record_colors[matched_i][1:], 
                            supplemental_counts = existing_record_counts[matched_i][1:]) 
        moon_queue.append(trace) 


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

            input_brightness = np.round(np.clip((watt/base_watt)*clip_ratio, clip_min, clip_max))
            animation_brightness = helper_classes.interpolate_stretched(frame_indices, input_brightness, time_arr)

            animation_plan[:, i] = animation_brightness
            animation_plan[:, i+1 : i+4] = input_pixels
        
        return animation_plan


    #length per step 
    animation_length = int(total_length*60/first_x*animation_fps) #10 minutes total? 

    tracks = [] #num = 5
    start_keyframe = None
    target_keyframe = None
    animation_plan = None

    full_animation = []

    for trace_i, trace in enumerate(moon_queue):
        print("record #", trace_i)
        if start_keyframe is None:
            init_tracks(trace, tracks)
            start_keyframe = generate_frame_from_trace_tracks(trace, tracks)
            print("start keyframe generated")
        target_keyframe = generate_frame_from_trace_tracks(trace, tracks)
        # print("animation length", animation_length, "animation_fps", animation_fps)
        progress = 0
        animation_plan = interpolate_two_frames(start_keyframe, target_keyframe, animation_length)
        start_keyframe = target_keyframe

        # print(animation_plan)
        while progress < len(animation_plan):
            f = animation_plan[progress]
            full_animation.append(f)
            frame = f[1:].reshape(-1, 4)
            print("progress--------", progress)
            if using_HD108:
                send_hd108_colors_with_brightness(frame)
                time.sleep(1/animation_fps)
            progress += 1
            
    full_animation = np.array(full_animation)
    np.savez(npz_file, animation_plan = full_animation)


if not recompute_animation:
    full_animation = np.load(npz_file)["animation_plan"]

if save_csv:
    csv_file = open(csv_path, "w", newline='') 
    writer = csv.writer(csv_file)

    human_header = ["frame"]
    for i in range(num_leds):
        human_header.extend([f"light_{i}_Wt", f"light_{i}_HEX" ])

    writer.writerows([human_header])

    def rgb_to_hex(r, g, b):
        return f" #{r:02x}{g:02x}{b:02x} "

    for i, f in enumerate(full_animation):
        frame = f[1:].reshape(-1, 4)
        
        row_data = [i]
        for led in frame:
            row_data.append(f" {led[0]:.2f}")
            row_data.append(rgb_to_hex(int(led[1])//256, int(led[2])//256, int(led[3])//256))

        writer.writerows([row_data])

if plot_stats:
    import matplotlib.pyplot as plt

    print(full_animation.shape)
    frames = full_animation[:, 1:]
    led_frames = frames.reshape(-1, 4)
    power_frames = np.floor(led_frames[:, 0])
    rgb_power_frames = np.sum(led_frames[:, 1:], axis = 1)/(2**16*3)*clip_max
    plt.figure()
    plt.hist([power_frames, rgb_power_frames], np.arange(0.5, clip_max+0.5, 1), label = ["power","rgb"])
    plt.title(f"distribution base={base_watt} max={clip_max}, min={clip_min}, ratio={clip_ratio} ")
    plt.legend()
    plt.show() 