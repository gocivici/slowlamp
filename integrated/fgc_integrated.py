import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os
from PIL import Image
from datetime import datetime
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
#import correct_color_HD108
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import helper_classes
import threading
import queue
import random
import colour

import cover

using_pi = True
day_length = 60 # minutes
animation_fps = 0.25 #inverse seconds
filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_file = open(f"{filename_time}_fgc_integrated.txt", "w")

stored_traces = []
if not diyVersion:
    trace_queue = queue.Queue()

using_HD108 = False

diyVersion = True

display_cv2_window = True

if using_pi:
    # import board
    # import neopixel
    # neopixels = neopixel.NeoPixel(board.D12, 24, brightness=0.2, pixel_order = neopixel.GRBW)

    # neopixel_grid = np.array([np.arange(8), 8+np.arange(8)[::-1], 16+np.arange(8)])

    #------------------------Camera Setup---------------------------------------
    from picamera2 import Picamera2
    tuning = Picamera2.load_tuning_file("/home/slowlamp3/Documents/slowlamp/image_proc/test.json") #imx477
    camera = Picamera2(tuning=tuning)
    # camera.resolution= (2028,1520)
    # camera.preview_configuration.main.format = "RGB888"
    # camera.set_controls({'AnalogueGain': 25.0, 'ExposureTime': 22000})
    # camera.start()

    # camera.start(show_preview=False)
    config = camera.create_still_configuration(main={"format": 'RGB888', "size":  (2028,1520)}, controls={"AwbEnable":0, "AwbMode": 3}) #DaylightMode=4, indoor=3
    camera.configure(config) 
    exposure_time = [3000000, 2000000, 1000000, 500000, 250000, 125000, 62500]

if using_HD108:
    import spidev

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

led_points = helper_classes.get_led_points_8cm()

num_leds = len(led_points)
base_watt = 70
num_per_group = num_leds//5

header = ["frame"]
for i in range(num_leds):
    header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B" ])

canvas_size = 500
feed_preview_size = 120
color_swatch_size = 40

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text

previous_img = None

def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)

def get_chroma(rgba): 
    # how saturated or vivid one color is 
    rgb = rgba[:3]
    rgb_scaled = sRGBColor(*[v / 255 for v in rgb], is_upscaled=False)
    lab = convert_color(rgb_scaled, LabColor)
    return (lab.lab_a ** 2 + lab.lab_b ** 2) ** 0.5

def resize_to_max(image, resize_max):
    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    return image

def capture_image():
    global using_pi
    if using_pi:
        frames = []
        for exp in exposure_time: 
            camera.set_controls({"ExposureTime": exp, "AnalogueGain": 1.0}) 
            camera.start()
            # file_name = f"frame_{exp}.jpg"
            frame_plain = camera.capture_array()
            frames.append(frame_plain)
            # cv2.imwrite(file_name, frame_plain)# no saving to memory 
            camera.stop()

        mergeMertens = cv2.createMergeMertens()
        fusion_res  = mergeMertens.process(frames)
        frame = np.clip(fusion_res * 255, 0, 255).astype('uint8')
    else:
        ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    
    return frame

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

def dominantColor(waitTime):
    # time.sleep(waitTime)
    global previous_img, stored_traces
    if previous_img is None:
        previous_img = capture_image()
        print("taking first picture")

    prev_preview = resize_to_max(previous_img, feed_preview_size)

    print("Waiting seconds:", waitTime)
    if waitTime > 0:
        time.sleep(waitTime) 

    # Get the current time in seconds since the epoch
    start_time_seconds = time.time()

    # Capture the new image
    current_img = capture_image()
    curr_preview = resize_to_max(current_img, feed_preview_size)

    print("taking second picture")
    
        
    current_img_bgr = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR )
    # cv2.imwrite('curim.png', current_img_bgr)
    # filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # cv2.imwrite(f'curim_{filename_time}.png', current_img_bgr)
    # cv2.imwrite('previm.png', previous_img)

#-------------------------Backgorund Substraction-----------------------------------
    # Compare with the previous image
    diff1=cv2.subtract(current_img,previous_img)
    diff2=cv2.subtract(previous_img,current_img)
    diff = diff1+diff2

    #adjustable threshold value original value =13
    diff[abs(diff)<13]=0

    #create mask based on threshold
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY) #rgb?
    gray[np.abs(gray) < 5] = 0
    fgmask = gray.astype(np.uint8)


    # morphological closing operation using ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


    #use the masks to extract the relevant parts from the image
    fgimg = cv2.bitwise_and(current_img,current_img,mask = morph)
    diff_preview = resize_to_max(fgimg, feed_preview_size)

    # cv2.imwrite('substract.png', fgimg)
    #-------------------------Color Detection-----------------------------------
    largest_color = np.array([255, 255, 255, 0])
    vibrant_color = np.array([255, 255, 255, 0])
    
    # image_rgb = cv2.cvtColor(fgimg,cv2.COLOR_BGR2RGB) #convert to RGB image
    image_rgb = fgimg
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    if len(imgNoB.flatten()) <= 10: #needs to be greater than K
        # <largest> Largest cluster in diff [same as now]
        # <expressible> Closest to expressible neopixel space in diff
        # <vibrant> Most vibrant color in diff 
        # <cluster> Largest change in cluster
        largest_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        vibrant_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        
        trace = helper_classes.Trace(vibrant_color, 0, traces_storing_mode="vaooo", supplemental_colors=[largest_color]*4, supplemental_counts = [0]*4)
        stored_traces.append(trace)
        storage_file.writelines([trace.print_trace()])
        if not diyVersion:
            trace_queue.put(trace)
        time_elapsed = time.time() - start_time_seconds  
        return (largest_color, 0), (vibrant_color, 0),  time_elapsed
    
    imgNoB = imgNoB[np.newaxis, :, :]
    # if color_mode == "lab":
    # print("lab mode")
    imgNoB = cv2.cvtColor(imgNoB, cv2.COLOR_RGB2LAB)

    imgNoB = np.float32(imgNoB) # convert to float. cv2.kmeans() requires to be in float 

    #Display the shape of pixel_values 
    # print(f"The shape of pixle_values is : {imgNoB.shape}")

    #Define the stopping creteria and number of clusters
    # stop after 10 iterations or when EPS(change in cluster centers) is less than 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    k = 5 # number of clusters
    previous_img = current_img
    #Apply the K-Means clustering
    DominantColors = np.array([[255, 255, 255, 0]] * k)
    label_counts_sorted = [1]*k
    
    try:
    # if True:
        ret, label, center = cv2.kmeans(imgNoB, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        label= np.reshape(label, label.size)
        label_counts = np.bincount(label)

        sorted_indices = np.argsort(-label_counts)

        # Sort center and label_counts accordingly
        center_sorted = center[sorted_indices]
        label_counts_sorted = label_counts[sorted_indices]

        # print("centers", list(enumerate(center)))
        for idx, color in enumerate(center_sorted):
            alpha = 255
            rgb_color = cv2.cvtColor(np.array([[color]]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0][0]
            DominantColors[idx] = list(rgb_color) + [alpha]  
            print(idx, color, rgb_color)
       
    except:
        vibrant_color = np.array([255, 255, 255, 0]).astype(np.uint8)
    
    
    vr_chosen_id = 0
    vr_count = 0
    max_chromma = 0
    for i, (count, rgba_color) in enumerate(zip(label_counts_sorted, DominantColors)):
        chroma = get_chroma(rgba_color)
        if chroma >= max_chromma:
            max_chromma = chroma
            vr_chosen_id = i
            vr_count = count

    vibrant_color = DominantColors[vr_chosen_id]
    largest_color = DominantColors[0]
    lg_count = label_counts_sorted[0]
    # print("selected vibrant color", vibrant_color, "with", vr_count)

    other_colors = []
    other_counts = []
    for i in range(0, len(DominantColors)):
        if i != vr_chosen_id:
            other_colors.append(DominantColors[i])
            other_counts.append(label_counts_sorted[i])
    
    trace = helper_classes.Trace(vibrant_color, vr_count,  
                  traces_storing_mode="vaooo", 
                    supplemental_colors = other_colors,
                    supplemental_counts = other_counts, )
    
    stored_traces.append(trace)
    storage_file.writelines([trace.print_trace()])
    storage_file.flush()
    if not diyVersion:
        trace_queue.put(trace)

    if display_cv2_window:
        # display some stuff
        canvas = np.ones((500, 800, 3)).astype(np.uint8)
        # if canvas_color is not None:
        #     canvas[:, :, 0] = canvas_color[2]
        #     canvas[:, :, 1] = canvas_color[1]
        #     canvas[:, :, 2] = canvas_color[0]
        
        canvas[0:prev_preview.shape[0], 0:prev_preview.shape[1], ::-1] = prev_preview
        canvas[0:curr_preview.shape[0], prev_preview.shape[1]:prev_preview.shape[1]+curr_preview.shape[1], ::-1] = curr_preview
        canvas[prev_preview.shape[0]:prev_preview.shape[0] + diff_preview.shape[0], 0:prev_preview.shape[1], ::-1] = diff_preview

        feed_preview_y = prev_preview.shape[0] + diff_preview.shape[0]
        feed_preview_x = 0
        for i, (count, center) in enumerate(zip(label_counts_sorted, DominantColors)):
            print("DominantColors (rgb) ", i, center)
            bgr_color = [center[2], center[1], center[0]]
            y_min = feed_preview_y + i*color_swatch_size
            canvas[y_min:y_min+color_swatch_size, feed_preview_x:feed_preview_x+color_swatch_size] = bgr_color
            canvas = cv2.putText(canvas, f'count {count}', (feed_preview_x+color_swatch_size, y_min+10), font, font_scale, text_color, thickness)
        
        circle_center = (190 + (800-feed_preview_size*2)//2, 250)
        vibrant_radius = 100
        ambient_radius = 190
        angle_per_swatch = 360/24
        for i, trace in enumerate(stored_traces):
            # swatch = trace.paint_trace()
            
            
            supp_colors = trace.supplemental_colors
            for j in range(len(supp_colors)):
                r = (ambient_radius-vibrant_radius)//len(supp_colors)*(len(supp_colors)-j) + vibrant_radius
                trace_ambient_color = to_cv2_color(supp_colors[j])
                canvas = cv2.ellipse(canvas,circle_center,(r, r), angle_per_swatch, i*angle_per_swatch, (i+1)*angle_per_swatch,
                                trace_ambient_color, -1)
            

            trace_vibrant_color = to_cv2_color(trace.main_color)
            canvas = cv2.ellipse(canvas,circle_center,(vibrant_radius, vibrant_radius), angle_per_swatch, i*angle_per_swatch, (i+1)*angle_per_swatch,
                                trace_vibrant_color, -1)
            
            if i >= 23:
                # clear "stored" traces used in diagonstic visualization
                stored_traces = []

        current_ambient_color = to_cv2_color(largest_color)
        current_vibrant_color = to_cv2_color(vibrant_color)

        if (lg_count + vr_count) == 0:
            vr_percent = 0.1
        else:
            vr_percent = vr_count/(lg_count+vr_count)
        
        canvas = cv2.circle(canvas, circle_center, vibrant_radius-40, current_ambient_color, -1)
        canvas = cv2.circle(canvas, circle_center, int((vibrant_radius-40)*vr_percent**(0.5)), current_vibrant_color, -1)
        
        cv2.imshow('frame', canvas)

    record = [tuple(vibrant_color[:-1]), vr_count,
            tuple(other_colors[0][:-1]), other_counts[0], tuple(other_colors[1][:-1]), other_counts[1],
            tuple(other_colors[2][:-1]), other_counts[2], tuple(other_colors[3][:-1]), other_counts[3], int(time.time()//3600)]
    print(record)
    cover.save(*record)

    time_elapsed = time.time() - start_time_seconds  
    return (largest_color, lg_count), (vibrant_color, vr_count), time_elapsed


def saveColor(color_array, filename = "archive.png"):
    color = tuple(map(int, color_array))
    # print(color)
    if os.path.exists(filename):
        img = Image.open(filename)
        existing_pixels = [px for px in img.getdata() if px != (0, 0, 0, 0)]
    else:
        existing_pixels = []

    # Append new color
    existing_pixels.append(color)
    # print(existing_pixels)

    # Set fixed number of columns
    columns = 24
    total_pixels = len(existing_pixels)

    # Calculate required number of rows
    rows = math.ceil(total_pixels / columns)

    # Pad pixel data with black pixels if necessary to fill the last row
    # padding = rows * columns - total_pixels
    # if padding > 0:
    #     existing_pixels.extend([(0, 0, 0)] * padding)

    # Create new image with calculated size
    img = Image.new("RGBA", (columns, rows))
    img.putdata(existing_pixels)
    img.save(filename)


def saveComposition(largest_color_count, vibrant_color_count, file_prefix = "composition_"):
    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = f"{file_prefix}{filename_time}.png"
    lg_color, lg_count = largest_color_count
    vr_color, vr_count = vibrant_color_count
    if (lg_count + vr_count) == 0:
        vr_percent = 0.1
    else:
        vr_percent = vr_count/(lg_count+vr_count)
    img_size = 100
    radius = int(np.sqrt(vr_percent*img_size**2/np.pi).round())
    canvas = np.ones((img_size, img_size, 4)).astype(np.uint8)
    canvas[:, :, 0] = lg_color[2]
    canvas[:, :, 1] = lg_color[1]
    canvas[:, :, 2] = lg_color[0]
    canvas[:, :, 3] = lg_color[3]
    print(vr_color, vr_count)
    canvas = cv2.circle(canvas, (img_size//2, img_size//2), radius, 
                        (int(vr_color[2]), int(vr_color[1]), int(vr_color[0]), int(vr_color[3])), thickness=-1)
    cv2.imwrite(image_file, canvas)

cap = None
if not using_pi:
    # Initialize the camera module
    cap = cv2.VideoCapture(0)

    # Check if the camera is initialized correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


def capture_thread_target():
    time_elapsed = 0 

    while True:
        largest_color, vibrant_color, time_elapsed = dominantColor(day_length*60-time_elapsed) #time in seconds
        print("color profile: largest_color, vibrant_color")
        print(largest_color, vibrant_color)
        # saveColor(largest_color[0], filename = "archive_largest.png")
        # saveColor(expressible_color, filename = "archive_expressible.png")
        # saveColor(vibrant_color[0], filename = "archive_vibrant.png")
        # saveColor(cluster_color, filename = "archive_cluster.png")
        # saveComposition(largest_color, vibrant_color, file_prefix = "composition_")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
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

        lab_full = [] 
        for flab in oklab_full:
            color_lab = colour.convert(flab, "Oklab", "CIE Lab")
            lab_full.append(color_lab)

        lab_full = np.array(lab_full)
        # print(lab_full[0])
        
        input_pixels = correct_color_HD108.correct_color_from_lab(lab_full)
        # print(input_pixels[0])

        input_brightness = np.clip((watt/base_watt)*10, 1, 10)
        animation_brightness = helper_classes.interpolate_stretched(frame_indices, input_brightness, time_arr)

        animation_plan[:, i] = animation_brightness
        animation_plan[:, i+1 : i+4] = input_pixels
    
    return animation_plan
    
def animation_thread_target():
    global trace_queue
    tracks = []
    start_keyframe = None
    target_keyframe = None
    progress = 0
    animation_length = 0
    animation_plan = None

    while True:
        if len(tracks) == 0 and not trace_queue.empty():
            if not diyVersion:
                trace = trace_queue.get()
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
            
            start_keyframe = generate_frame_from_trace_tracks(trace, tracks)
            print("start keyframe generated")
        else:
            if trace_queue.empty(): 
                if animation_plan is None:
                    time.sleep(1/animation_fps*10) #actually waiting for capturing 
                else:
                    # move to the next animation frame
                    if progress < len(animation_plan):
                        f = animation_plan[progress]
                        frame = f[1:].reshape(-1, 4)/3
                        if using_HD108:
                            send_hd108_colors_with_brightness(frame)
                        print("----------", progress)
                        progress += 1
                    time.sleep(1/animation_fps)
            else: # has new capture
                if target_keyframe is not None:
                    # bridge the old one 
                    remaining_animation = animation_plan[progress:]
                    for f in remaining_animation:
                        frame = f[1:].reshape(-1, 4)/3
                        if using_HD108:
                            send_hd108_colors_with_brightness(frame)
                        
                        print("---------- rushing")
                        time.sleep(1/15) #let's speed up a bit... 
                
                    start_keyframe = target_keyframe
                if not diyVersion:
                    trace = trace_queue.get()
                target_keyframe = generate_frame_from_trace_tracks(trace, tracks)
                animation_length = int((target_keyframe[0] - start_keyframe[0])*animation_fps*0.8) # 0.8 = a pause for computation time
                print("animation length", animation_length)
                progress = 0
                animation_plan = interpolate_two_frames(start_keyframe, target_keyframe, animation_length)        
        
# capture_thread = threading.Thread(target=capture_thread_target) # QObject::startTimer: Timers cannot be started from another thread
if not diyVersion:
    animation_thread = threading.Thread(target=animation_thread_target, daemon=True)

# capture_thread.start()
if not diyVersion:
    animation_thread.start()

# capture_thread.join()
# animation_thread.join()

if not diyVersion:
    capture_thread_target()

# animation_thread.join()

if not diyVersion:
    print("Main: All threads finished.")

if not using_pi:
    # Release the camera resources
    cap.release()
# else:
    # neopixels.fill( (0, 0, 0))

if display_cv2_window:
    cv2.destroyAllWindows()

if using_HD108:
    colors_16bit = [(1, 0, 0, 0)]*num_leds
    send_hd108_colors_with_brightness(colors_16bit)
    send_hd108_colors_with_brightness(colors_16bit)
    spi.close()

storage_file.flush()
storage_file.close()
