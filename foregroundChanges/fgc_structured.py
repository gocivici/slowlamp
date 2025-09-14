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
# import correct_color_RGBW
from scipy.spatial.distance import cdist



using_pi = True
pixel_mode = "daily" # "reactive" or "daily"
traces_storing_mode = "complementary" # "single", "complementary", or "neighbor"
display_matrix_mode = "gradient"
day_length =  0.4 #minutes
filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_file = open(f"{filename_time}_fgc_yumeng.txt", "w")

if using_pi:
    # import board
    # import neopixel
    # neopixels = neopixel.NeoPixel(board.D12, 24, brightness=0.2, pixel_order = neopixel.GRBW)

    # neopixel_grid = np.array([np.arange(8), 8+np.arange(8)[::-1], 16+np.arange(8)])

    #------------------------Camera Setup---------------------------------------
    from picamera2 import Picamera2
    camera = Picamera2()
    camera.resolution= (2028,1520)
    camera.preview_configuration.main.format = "RGB888"
    camera.set_controls({'AnalogueGain': 25.0, 'ExposureTime': 22000})
    camera.start()

    camera.start(show_preview=False)
    
canvas_size = 500
feed_preview_size = 120
color_swatch_size = 40

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text

previous_img = None

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

class Trace:
    def __init__(self, main_color, count, position, traces_storing_mode="single", 
                 supplemental_colors=None, supplemental_counts=None, supplemental_positions=None):
        self.main_color = np.array(main_color[:3])
        self.main_count = count
        self.main_pos = position
        self.traces_storing_mode = traces_storing_mode
        self.supplemental_colors = []
        if traces_storing_mode != "single":
            self.supplemental_colors = np.array(supplemental_colors)[:, :3]
            self.supplemental_counts = supplemental_counts
            self.supplemental_pos = supplemental_positions
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def print_trace(self):
        main_str = f'mode: {self.traces_storing_mode}; \
main_color (rgb) #{self.main_count}: <{int(self.main_pos[0])},{int(self.main_pos[1])}>({int(self.main_color[0])}, {int(self.main_color[1])}, {int(self.main_color[2])});'
        supp_str = ''
        if self.traces_storing_mode != "single":
            supp_str = "supplemental_colors: ["
            for color, count, pos in zip(self.supplemental_colors, self.supplemental_counts, self.supplemental_pos):
                supp_str += f' #{count}  <{int(pos[0])},{int(pos[1])}>({int(color[0])}, {int(color[1])}, {int(color[2])}),'
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
    
def linear_gradient(rgbw1, rgbw2, ratio):
    r = int(rgbw2[0]*(1-ratio) + rgbw1[0]*ratio)
    g = int(rgbw2[1]*(1-ratio) + rgbw1[1]*ratio)
    b = int(rgbw2[2]*(1-ratio) + rgbw1[2]*ratio)
    return (r, g, b, 0)

def capture_image():
    global using_pi
    if using_pi:
        frame = camera.capture_array()
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

    print("Waiting seconds:",waitTime)
    if waitTime > 0:
        time.sleep(waitTime) 

    # Get the current time in seconds since the epoch
    start_time_seconds = time.time()

    # Capture the new image
    current_img = capture_image()
    curr_preview = resize_to_max(current_img, feed_preview_size)

    print("taking second picture")
    
        
    current_img_bgr = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR )
    cv2.imwrite('curim.png', current_img_bgr)
    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f'curim_{filename_time}.png', current_img_bgr)
    cv2.imwrite('previm.png', previous_img)

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

    cv2.imwrite('substract.png', fgimg)
    #-------------------------Color Detection-----------------------------------
    largest_color = np.array([255, 255, 255, 0])
    expressible_color = np.array([255, 255, 255, 0])
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
        expressible_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        vibrant_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        
        trace = Trace(vibrant_color, 0, (0, 0), traces_storing_mode="vaooo", supplemental_colors=[largest_color]*4, supplemental_counts = [0]*4, supplemental_positions=[[0, 0]]*4)
        stored_traces.append(trace)
        storage_file.writelines([trace.print_trace()])
        time_elapsed = time.time() - start_time_seconds  
        return (largest_color, 0),  (vibrant_color, 0), time_elapsed
    
    imgNoB = imgNoB[np.newaxis, :, :]
    # if color_mode == "lab":
    # print("lab mode")
    imgNoB = cv2.cvtColor(imgNoB, cv2.COLOR_RGB2LAB)

    imgNoB = np.float32(imgNoB) # convert to float. cv2.kmeans() requires to be in float 

    #Display the shape of pixel_values 
    print(f"The shape of pixle_values is : {imgNoB.shape}")

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

        print("centers", list(enumerate(center)))
        for idx, color in enumerate(center_sorted):
            alpha = 255
            rgb_color = cv2.cvtColor(np.array([[color]]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0][0]
            DominantColors[idx] = list(rgb_color) + [alpha]  
            print(idx, color, rgb_color)

        fgLAB = cv2.cvtColor(fgimg, cv2.COLOR_RGB2LAB)
        distances = cdist(center_sorted, fgLAB.reshape(-1, 3))
        sorted_indices = np.argsort(distances, axis=0)
        # Initialize sums
        center_of_masses_x = np.zeros(len(center_sorted))
        center_of_masses_y = np.zeros(len(center_sorted))
        center_of_masses_count = np.zeros(len(center_sorted))
        for y in range(fgLAB.shape[0]):
            for x in range(fgLAB.shape[1]):
                if np.all(fgimg[y, x] <= 0):
                    continue
                j = y * fgLAB.shape[1] + x
                assigned_center = sorted_indices[0][j]
                center_of_masses_x[assigned_center] += x
                center_of_masses_y[assigned_center] += y
                center_of_masses_count[assigned_center] += 1

        # Avoid division by zero
        valid = center_of_masses_count > 0
        center_of_masses_x[valid] /= center_of_masses_count[valid]
        center_of_masses_y[valid] /= center_of_masses_count[valid]
        print(center_of_masses_x, center_of_masses_y)
    except:
        vibrant_color = np.array([255, 255, 255, 0]).astype(np.uint8)
        center_of_masses_x = np.zeros(k)
        center_of_masses_y = np.zeros(k)
    
    
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
    lg_count = label_counts_sorted[0]
    print("selected vibrant color", vibrant_color, "with", vr_count)

    other_colors = []
    other_counts = []
    other_pos = []
    for i in range(0, len(DominantColors)):
        if i != vr_chosen_id:
            other_colors.append(DominantColors[i])
            other_counts.append(label_counts_sorted[i])
            other_pos.append([center_of_masses_x[i], center_of_masses_y[i]])
    trace = Trace(vibrant_color, vr_count, 
                 [center_of_masses_x[vr_chosen_id], center_of_masses_y[vr_chosen_id]], 
                  traces_storing_mode="vaooo", 
                    supplemental_colors = other_colors,
                    supplemental_counts = other_counts, 
                    supplemental_positions = other_pos)
    stored_traces.append(trace)
    storage_file.writelines([trace.print_trace()])
    storage_file.flush()

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
    vibrant_radius = 150
    ambient_radius = 190
    angle_per_swatch = 360/24
    for i, trace in enumerate(stored_traces):
        # swatch = trace.paint_trace()
        trace_vibrant_color = to_cv2_color(trace.main_color)
        trace_ambient_color = to_cv2_color(trace.supplemental_colors[0])
        canvas = cv2.ellipse(canvas,circle_center,(ambient_radius, ambient_radius), angle_per_swatch, i*angle_per_swatch, (i+1)*angle_per_swatch,
                             trace_ambient_color, -1)
        canvas = cv2.ellipse(canvas,circle_center,(vibrant_radius, vibrant_radius), angle_per_swatch, i*angle_per_swatch, (i+1)*angle_per_swatch,
                             trace_vibrant_color, -1)
        if i >= 23:
            # clear "stored" traces used in diagonstic visualization
            stored_traces = []

    # current_ambient_color = to_cv2_color(largest_color)
    # current_vibrant_color =to_cv2_color(vibrant_color)

    # if (lg_count + vr_count) == 0:
    #     vr_percent = 0.1
    # else:
    #     vr_percent = vr_count/(lg_count+vr_count)
    
    # canvas = cv2.circle(canvas, circle_center, vibrant_radius-40, current_ambient_color, -1)
    inner_r = vibrant_radius-40
    canvas = cv2.circle(canvas, circle_center, inner_r, (0, 0, 0), -1)
    x0 = circle_center[0] - inner_r/np.sqrt(2)
    y0 = circle_center[1] - inner_r/np.sqrt(2)
    h, w = fgimg.shape[:-1]
    s = inner_r/np.sqrt(2)*2
    for  i, (count, rgba_color) in enumerate(zip(label_counts_sorted, DominantColors)):
        bgr_color = (int(rgba_color[2]), int(rgba_color[1]), int(rgba_color[0]))
        center_x = int(center_of_masses_x[i]/w*s+x0)
        center_y = int(center_of_masses_y[i]/h*s+y0)
        canvas = cv2.circle(canvas, (center_x, center_y), int(count/h/w*60), bgr_color, -1)

    # canvas = cv2.circle(canvas, circle_center, int((vibrant_radius-40)*vr_percent**(0.5)), current_vibrant_color, -1)
    
    cv2.imshow('frame', canvas)

    if using_pi:
        # for row, trace in enumerate(stored_traces[::-1]):
        #     if row >= len(neopixel_grid[0]):
        #         break
        #     for i in neopixel_grid[:, row]:
        #         neopixels[i] = trace.supplemental_colors[0]
        # neopixels.fill(correct_color_RGBW.correct_color(vibrant_color))
        pass

        # for row in range(4):
        #     for i in neopixel_grid[:, row]:
        #         neopixels[i] = vibrant_color

    # <largest> Largest cluster in diff [same as now]
    # <expressible> Closest to expressible neopixel space in diff
    # <vibrant> Most vibrant color in diff 
    # <cluster> Largest change in cluster

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
    print(existing_pixels)

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

tracks = [] #rgb
canvas_color = None
memory_color = None
stored_traces = []

time_elapsed = 0 
while True:
    largest_color, vibrant_color, time_elapsed = dominantColor(day_length*60-time_elapsed) #time in seconds
    print("color profile: largest_color, expressible_color, vibrant_color, cluster_color")
    print(largest_color, vibrant_color)
    saveColor(largest_color[0], filename = "archive_largest.png")
    # saveColor(expressible_color, filename = "archive_expressible.png")
    saveColor(vibrant_color[0], filename = "archive_vibrant.png")
    # saveColor(cluster_color, filename = "archive_cluster.png")
    # saveComposition(largest_color, vibrant_color, file_prefix = "composition_")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not using_pi:
    # Release the camera resources
    cap.release()
# else:
    # neopixels.fill( (0, 0, 0))
    
cv2.destroyAllWindows()
storage_file.flush()
storage_file.close()
