import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os
import subprocess
from PIL import Image
from datetime import datetime
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import threading
import queue
import random
import colour

import helper_classes
import json


def loadConfig(pathtofile):
    with open(pathtofile, 'r') as file:
            config = json.load(file)
            return config

#load config data json format
configData = loadConfig('camera_perception/config.json')


using_pi = configData.get("using_pi")
day_length = configData.get("day_length") # minutes
animation_fps = configData.get("animation_fps") #inverse seconds
using_HD108 = False
has_animation = False
diyVersion = False
display_cv2_window = True
fast_stream_interval = configData.get("fast_stream_interval")*60

filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_file = open(f"{filename_time}_fgc_integrated.txt", "w")

stored_traces = []
trace_queue = queue.Queue()

if using_pi:
    # import board
    # import neopixel
    # neopixels = neopixel.NeoPixel(board.D12, 24, brightness=0.2, pixel_order = neopixel.GRBW)

    # neopixel_grid = np.array([np.arange(8), 8+np.arange(8)[::-1], 16+np.arange(8)])

    #------------------------Camera Setup---------------------------------------
    from picamera2 import Picamera2
    if not diyVersion:
        tuning_file_path = configData.get("camera_tuning_file")
        if tuning_file_path != "":
            tuning = Picamera2.load_tuning_file(tuning_file_path) #imx477
            camera = Picamera2(tuning=tuning)
        else: 
            camera = Picamera2()
        # exposure_time = [3000000, 2000000, 1000000, 500000, 250000, 125000, 62500]
        exposure_time = [1000000, 250000, 62500] #for testing
    else:
        camera = Picamera2()
        exposure_time = [62500]

    config = camera.create_still_configuration(main={"format": 'RGB888', "size":  (2028,1520)}, controls={"AwbEnable":0, "AwbMode": 3}) #DaylightMode=4, indoor=3
    camera.configure(config) 

    # camera.resolution= (2028,1520)
    # camera.preview_configuration.main.format = "RGB888"
    # camera.set_controls({'AnalogueGain': 25.0, 'ExposureTime': 22000})
    # camera.start()

    # camera.start(show_preview=False)

cap = None
if not using_pi:
    # Initialize the camera module
    cap = cv2.VideoCapture(0)

    # Check if the camera is initialized correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

canvas = None
canvas_size = 500
feed_preview_size = 120
color_swatch_size = 80

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text

previous_img = None


color_max_distance = 12

trace_length_min = 2
trace_length_max = 5
max_frame_number = 12
fast_tracks = []


def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*rgb) #is_upscaled=True expects 0~255
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
            camera.start(show_preview=False)
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

def clean_exit():
    if not using_pi:
    # Release the camera resources
        cap.release()
    # else:
        # neopixels.fill( (0, 0, 0))

    if display_cv2_window:
        cv2.destroyAllWindows()

    storage_file.flush()
    storage_file.close()
    exit()

def smart_delay(waitTime_seconds):
    """
    Waits for waitTime_seconds, but breaks early if 'c' is pressed.
    Exits the program if 'q' is pressed,
    Returns True if 'c' was pressed, False if it timed out.
    """

    global display_cv2_window 

    start_time = time.time()

    if display_cv2_window:
    
        while (time.time() - start_time) < waitTime_seconds:
            # 1. Take a 1ms break to check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # 2. Check if the 'c' key was pressed
            if key == ord('c'):
                print("User triggered capture early!")
                return True
                
            # 3. (Optional) Check if 'q' was pressed to exit the app entirely
            if key == ord('q'):
                clean_exit()
    else: 
        if waitTime_seconds > 0:
            time.sleep(waitTime_seconds)

    return False

def display_diagnostic_at_main(canvas, prev_preview, curr_preview):
    global stored_traces
    # display some stuff
    # canvas = np.ones((500, 800, 3)).astype(np.uint8)
    # if canvas_color is not None:
    #     canvas[:, :, 0] = canvas_color[2]
    #     canvas[:, :, 1] = canvas_color[1]
    #     canvas[:, :, 2] = canvas_color[0]
    
    canvas[0:prev_preview.shape[0], 0:prev_preview.shape[1], ::-1] = prev_preview
    canvas[0:curr_preview.shape[0], prev_preview.shape[1]:prev_preview.shape[1]+curr_preview.shape[1], ::-1] = curr_preview
    
    cv2.imshow('frame', canvas)

    return canvas

def display_diagnostic_at_fast_stream(canvas, frame, new_colors, candidate_color = None):
    global fast_tracks
    feed_preview = resize_to_max(frame, feed_preview_size)
    feed_preview_y = feed_preview.shape[0]
    feed_preview_x = feed_preview.shape[1]
    canvas[0:feed_preview_y, feed_preview_x:feed_preview_x*2, ::-1] = feed_preview
    if candidate_color is not None:
        print(candidate_color)
        canvas[feed_preview_y:feed_preview_y*2, feed_preview_x:feed_preview_x*2, ::-1] = candidate_color[0][:-1]

    for ci in range(len(new_colors)): #lab
        # start at feed_preview
        rgb_color = new_colors[ci]
        # print(lab_color)
        # rgb_color = lab_color #lab_to_rgb(lab_color)
        y_min = feed_preview_y*2
        x_min = ci * color_swatch_size
        canvas[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size, 0] = rgb_color[2]
        canvas[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size, 1] = rgb_color[1]
        canvas[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size, 2] = rgb_color[0]

    for i, track in enumerate(fast_tracks):
        rgb_color, count_str, age_str, missed_str = track.print_swatch()
        bgr_color = [rgb_color[2], rgb_color[1], rgb_color[0]]
        y_min = feed_preview_y*2 + 2*color_swatch_size
        x_min = i*color_swatch_size
        canvas[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size] = bgr_color
        # canvas = cv2.putText(canvas, f'age{age_str}', (feed_preview_x+color_swatch_size, y_min+10), font, font_scale, text_color, thickness)
        # canvas = cv2.putText(canvas, f'missed{missed_str}', (feed_preview_x+color_swatch_size, y_min+25), font, font_scale, text_color, thickness)

    cv2.imshow('frame', canvas)

    return canvas


def dominantColor(waitTime):
    # time.sleep(waitTime)
    global previous_img, stored_traces, canvas, fast_tracks

    if previous_img is None:
        previous_img = capture_image()
        print("taking first picture")


    fast_vr_color = [0, 0, 0, 255]
    fast_count = 0
            
    print("Waiting seconds or a 'c' key press:", waitTime)
    if waitTime > 0:
        num_cycles = int(waitTime/fast_stream_interval) 
        fast_stream_start_time = time.time()
        # remaining_time = waitTime - (fast_stream_interval*num_cycles)
        if num_cycles > 0:
            candidates = []
            interval_wait_time = fast_stream_interval

            for fast_cycle_index in range(num_cycles):

                smart_delay(interval_wait_time)
                fast_interval_start_time = time.time()
                curr_between_img = capture_image()
                new_colors, label_counts = helper_classes.extract_colors_cv2(curr_between_img, 5)
                n, m = len(fast_tracks), len(new_colors)
                cost_matrix = np.full((n, m), np.inf)

                # Build cost matrix
                for i, track in enumerate(fast_tracks):
                    for j, color in enumerate(new_colors):
                        track_lab_color = rgb_to_lab(track.color)
                        new_lab_color = rgb_to_lab(color)
                        dist = helper_classes.ciede2000(track_lab_color, new_lab_color)
                        # print(dist)
                        cost_matrix[i, j] = dist

                # row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # match color to closest existing track, not a global optimization 
                if len(cost_matrix) > 0:
                    closest_track_ids = np.argmin(cost_matrix, axis=0)
                else:
                    closest_track_ids = []

                assigned_tracks = set()
                assigned_colors = set()

                for ni, ti in enumerate(closest_track_ids):
                    if cost_matrix[ti, ni] < color_max_distance:
                        fast_tracks[ti].update(new_colors[ni], label_counts[ni])
                        assigned_tracks.add(ti)
                        assigned_colors.add(ni)

                # Age and remove unmatched tracks
                new_tracks = []
                new_candidate_detected = False
                for i, track in enumerate(fast_tracks):
                    if i not in assigned_tracks:
                        track.missed += 1
                    if track.missed <= max_frame_number: # max_missed:
                        new_tracks.append(track)
                    if track.age >= trace_length_min and track.age <= trace_length_max and track.missed == trace_length_min:
                        # canvas_color = track.color #rgb
                        new_candidate_detected = True
                        candidates.append((track.color, track.count))

                # Add new tracks for unmatched colors
                for j, color in enumerate(new_colors):
                    if j not in assigned_colors:
                        new_tracks.append(helper_classes.FastTrack(color, label_counts[j]))

                fast_tracks = new_tracks


                if display_cv2_window:
                    if len(candidates) > 0:
                        canvas = display_diagnostic_at_fast_stream(canvas, curr_between_img, new_colors, candidates[-1])
                    else:
                        canvas = display_diagnostic_at_fast_stream(canvas, curr_between_img, new_colors)


                interval_wait_time = fast_stream_interval - (time.time()-fast_interval_start_time)
            # end for fast_cycle_index

            if len(candidates) > 0:
                fast_vr_color = [0, 0, 0, 255]
                fast_count = 0
                max_chromma = 0
                for (rgb_color, count) in candidates:
                    rgba_color = [rgb_color[0], rgb_color[1], rgb_color[2], 255]
                    chroma = get_chroma(rgba_color)
                    if chroma >= max_chromma:
                        max_chromma = chroma
                        fast_vr_color = rgba_color
                        fast_count = count

                print("Captured trace!", fast_vr_color, fast_count)
            else:
                print("No trace captured!!!")

        # end if num_cycles > 0
        remaining_time = waitTime - (time.time() - fast_stream_start_time)
        smart_delay(remaining_time)
        
    # Get the current time in seconds since the epoch
    start_time_seconds = time.time()

    if display_cv2_window:
        canvas = np.ones((500, 800, 3)).astype(np.uint8)

    prev_preview = resize_to_max(previous_img, feed_preview_size)

    # Capture the new image
    current_img = capture_image()
    curr_preview = resize_to_max(current_img, feed_preview_size)

    if display_cv2_window:
        canvas = display_diagnostic_at_main(canvas, prev_preview, curr_preview)


    previous_img = current_img

    time_elapsed = time.time() - start_time_seconds  
    return time_elapsed

def capture_thread_target():
    global canvas
    if display_cv2_window:
        # display some stuff
        canvas = np.ones((500, 800, 3)).astype(np.uint8)
        cv2.imshow('frame', canvas)

    time_elapsed = 0 

    while True:
        time_elapsed = dominantColor(day_length*60-time_elapsed) #time in seconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
capture_thread_target()

print("Main: All threads finished.")

clean_exit()
