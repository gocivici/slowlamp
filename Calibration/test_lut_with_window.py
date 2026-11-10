import numpy as np
import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os

from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
# import correct_color_RGBW
import board
from integrated.fgc_double_stream import get_colors_from_difference
import spidev
import json
import kmeans_lab_medoids_spread

feed_preview_size = 120
using_pi = True 

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

def capture_image():
    global using_pi
    if using_pi:
        frames = []
        camera.start(show_preview=False)
        try: 
            for exp in exposure_time: 
                camera.set_controls({"ExposureTime": exp, "AnalogueGain": 1.0}) 
                # file_name = f"frame_{exp}.jpg"
                time.sleep(0.05)
                frame_plain = camera.capture_array()
                frames.append(frame_plain)
                # cv2.imwrite(file_name, frame_plain)# no saving to memory 
        finally:
            camera.stop()

        mergeMertens = cv2.createMergeMertens()
        fusion_res  = mergeMertens.process(frames)
        frame = np.clip(fusion_res * 255, 0, 255).astype('uint8')
    else:
        pass
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    
    return frame

def to_cv2_color(color):
    return (int(color[2]), int(color[1]), int(color[0]))

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

def get_colors_from_capture(current_img, num_colors, separate_vibrant = True):
    #-------------------------Backgorund Substraction-----------------------------------
    # Compare with the previous image
    # diff1=cv2.subtract(current_img, previous_img)
    # diff2=cv2.subtract(previous_img, current_img)
    # diff = diff1+diff2

    # #adjustable threshold value original value =13
    # diff[abs(diff)<13]=0

    # #create mask based on threshold
    # gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY) #rgb?
    # gray[np.abs(gray) < 5] = 0
    # fgmask = gray.astype(np.uint8)

    # # morphological closing operation using ellipse
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # morph = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


    # #use the masks to extract the relevant parts from the image
    # fgimg = cv2.bitwise_and(current_img,current_img,mask = morph)
    diff_preview = resize_to_max(current_img, feed_preview_size)

    # cv2.imwrite('substract.png', fgimg)
    #-------------------------Color Detection-----------------------------------
    largest_color = np.array([255, 255, 255, 0])
    vibrant_color = np.array([255, 255, 255, 0])
    
    # image_rgb = cv2.cvtColor(fgimg,cv2.COLOR_BGR2RGB) #convert to RGB image
    image_rgb = current_img
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    if len(imgNoB.flatten()) <= num_colors * kmeans_lab_medoids_spread.OVERSAMPLE_FACTOR*3: 
        largest_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        vibrant_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        if not separate_vibrant:
            return diff_preview, [largest_color]*(num_colors), [0]*(num_colors)

        return diff_preview, vibrant_color, 0, [largest_color]*(num_colors-1), [0]*(num_colors-1)


    DominantColors, label_counts_sorted = kmeans_lab_medoids_spread.extract_palette(imgNoB, k=num_colors)
    
    k = num_colors # number of clusters
    # previous_img = current_img
    #Apply the K-Means clustering

    if not separate_vibrant:
        return diff_preview, DominantColors, label_counts_sorted

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

    return diff_preview, vibrant_color, vr_count, other_colors, other_counts

with open("slow_lamp_calibration.json") as f:
    cal = json.load(f)


# --- Option A: full LUT, direct lookup, ~1.5MB in memory ---

r_full = cal["luts_full"]["red"]
g_full = cal["luts_full"]["green"]
b_full = cal["luts_full"]["blue"]


def correct_full(r, g, b):
    """Direct lookup, no math at runtime."""
    return r_full[r], g_full[g], b_full[b]

# --- must match your calibration config ---
NUM_LEDS = 436
GLOBAL_BRIGHTNESS = 6
DIMMEST_STEP = 10       # lowest non-zero drive value in your sweep


def send_hd108_colors(colors_16bit, global_brightness=GLOBAL_BRIGHTNESS):
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 2000000
    data = []
    data.extend([0x00] * 8)
    for r16, g16, b16 in colors_16bit:
        if global_brightness <= 0:
            brightness = 0
        else:
            brightness = ((1 << 15)
                          | (global_brightness << 10)
                          | (global_brightness << 5)
                          | global_brightness)
        data.extend([
            (brightness >> 8) & 0xFF, brightness & 0xFF,
            (r16 >> 8) & 0xFF, r16 & 0xFF,
            (g16 >> 8) & 0xFF, g16 & 0xFF,
            (b16 >> 8) & 0xFF, b16 & 0xFF,
        ])
    num_end = 2 * (len(colors_16bit) + 1)
    data.extend([0xFF] * num_end)
    spi.writebytes2(data)
    spi.close()


def fill_all(r, g, b):
    colors = [(int(r), int(g), int(b))] * NUM_LEDS
    send_hd108_colors(colors)
    send_hd108_colors(colors)  

for r in list(range(0, 255, 64))+[255]:
    for g in list(range(0, 255, 64))+[255]:
        for b in list(range(0, 255, 64))+[255]:

            current_img = capture_image()
            needed_clusters = 5
            prev_preview, vibrant_color, vr_count, supplemental_colors, supplemental_counts = get_colors_from_difference(current_img, needed_clusters)


            #target_colors.append( (r, g, b))
            if supplemental_counts[0] > vr_count:
                target_rgb = supplemental_colors[0]
            else:
                target_rgb = vibrant_color
    
            print("target_rgb", target_rgb)
            
            # correct_rgbw =  correct_color_RGBW.correct_color(target_rgb)
            # print("correct_rgbw", correct_rgbw)

            fill_all(target_rgb[0]*255, target_rgb[1]*255, target_rgb[2]*255)

            canvas = np.ones((500, 500, 3)).astype(np.uint8)
            canvas[:, :, 0] = target_rgb[2]
            canvas[:, :, 1] = target_rgb[1]
            canvas[:, :, 2] = target_rgb[0]

            canvas[0:prev_preview.shape[0], 0:prev_preview.shape[1], ::-1] = prev_preview

            cv2.imshow('frame', canvas)

            res = cv2.waitKey(0)

            fill_all(*correct_full(target_rgb[0]*255, target_rgb[1]*255, target_rgb[2]*255))

            res = cv2.waitKey(0)
            
            if res == ord('q'):
                break
                
        if res == ord('q'):
            break
            
    if res == ord('q'):
        break
                
fill_all(0, 0, 0)
