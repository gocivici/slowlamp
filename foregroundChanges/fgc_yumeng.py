import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os
from PIL import Image
from datetime import datetime


using_pi = True
pixel_mode = "daily" # "reactive" or "daily"
traces_storing_mode = "complementary" # "single", "complementary", or "neighbor"
display_matrix_mode = "gradient"
day_length = 5 #minutes
filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_file = open(f"{filename_time}_fgc_yumeng.txt", "w")

if using_pi:
    import board
    import neopixel
    neopixels = neopixel.NeoPixel(board.D12, 24, brightness=0.2, pixel_order = neopixel.GRBW)

    neopixel_grid = np.array([np.arange(8), 8+np.arange(8)[::-1], 16+np.arange(8)])

    #------------------------Camera Setup---------------------------------------
    from picamera2 import Picamera2
    camera = Picamera2()
    camera.resolution= (2028,1520)
    camera.preview_configuration.main.format = "RGB888"
    camera.set_controls({'AnalogueGain': 25.0, 'ExposureTime': 22000})
    camera.start()

    camera.start(show_preview=False)
    
canvas_size = 500
feed_preview_size = 100
color_swatch_size = 50

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text

previous_img = None


def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

class Trace:
    def __init__(self, main_color, traces_storing_mode="single", supplemental_colors=None):
        self.main_color = np.array(main_color)
        self.traces_storing_mode = traces_storing_mode
        self.supplemental_colors = []
        if traces_storing_mode != "single":
            self.supplemental_colors = np.array(supplemental_colors)
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def print_trace(self):
        main_str = f'mode: {self.traces_storing_mode}; main_color (rgb): ({int(self.main_color[0])}, {int(self.main_color[1])}, {int(self.main_color[2])});'
        supp_str = ''
        if self.traces_storing_mode != "single":
            supp_str = "supplemental_colors: ["
            for color in self.supplemental_colors:
                supp_str += f'({int(color[0])}, {int(color[1])}, {int(color[2])}),'
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

def dominantColor(waitTime):
    # time.sleep(waitTime)
    global previous_img
    if previous_img is None:
        previous_img = capture_image()
        print("taking first picture")

    prev_preview = resize_to_max(previous_img, feed_preview_size)

    print("Waiting seconds:",waitTime)
    time.sleep(waitTime) 


    # Capture the new image
    current_img = capture_image()
    curr_preview = resize_to_max(current_img, feed_preview_size)

    print("taking second picture")

    cv2.imwrite('curim.png', current_img)
    cv2.imwrite('previm.png', previous_img)

    # current_img = cv2.resize(current_img, (480, 360))

#-------------------------Backgorund Substraction-----------------------------------
    # Compare with the previous image
    diff1=cv2.subtract(current_img,previous_img)
    diff2=cv2.subtract(previous_img,current_img)
    diff = diff1+diff2


    # cv2.imshow("previous", previous_img)
    # cv2.imshow("current", current_img)
    # cv2.waitKey(1)  # Refresh the window

    

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

    # image_rgb = cv2.cvtColor(fgimg,cv2.COLOR_BGR2RGB) #convert to RGB image
    image_rgb = fgimg
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    if len(imgNoB.flatten()) <= 10: #needs to be greater than K
        return np.array([255, 255, 255, 0])
    
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
    DominantColors = np.array([[0, 0, 0, 0]] * k)

    try:
        ret, label, center = cv2.kmeans(imgNoB, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        label= np.reshape(label, label.size)
        label_counts = np.bincount(label)

        sorted_indices = np.argsort(-label_counts)

        # Sort center and label_counts accordingly
        center_sorted = center[sorted_indices]
        label_counts_sorted = label_counts[sorted_indices]

        print("centers", list(enumerate(center)))
        for idx, color in enumerate(center_sorted):
            # plt.subplot(1, k, idx+1)
            # plt.axis('off')
            # plt.imshow([[color / 255]])  # Normalize RGB values to range [0,1]
            # plt.title(f'%{(100*label_counts_sorted[idx]/imgNoB.shape[0]):.2f}')
            # alpha = 255 * label_counts_sorted[idx] / imgNoB.shape[0]
            alpha = 255
            rgb_color = cv2.cvtColor(np.array([[color]]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0][0]
            DominantColors[idx] = list(rgb_color) + [alpha]  
            print(idx, color, rgb_color)
        print("selected dominant color", DominantColors[0])


        # display some stuff
        canvas = np.ones((500, 500, 3)).astype(np.uint8)
        if canvas_color is not None:
            canvas[:, :, 0] = canvas_color[2]
            canvas[:, :, 1] = canvas_color[1]
            canvas[:, :, 2] = canvas_color[0]
        
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
        
        # if using_pi:
        #    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) #why??
            
        cv2.imshow('frame', canvas)

        
    except:
        DominantColors[0]=[255, 255, 255, 0]

    trace = Trace(list(DominantColors[0].round()))
    stored_traces.append(trace)
    storage_file.writelines([trace.print_trace()])

    if using_pi:
        for row, trace in enumerate(stored_traces[::-1]):
            if row >= len(neopixel_grid[0]):
                break
            for i in neopixel_grid[:, row]:
                neopixels[i] = trace.main_color

    return DominantColors[0].round()


def saveColor(color_array):
    color = tuple(map(int, color_array))
    # print(color)
    if os.path.exists("archive.png"):
        img = Image.open("archive.png")
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
    img.save("archive.png")



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

while True:
    color = dominantColor(day_length*60) #time in seconds
    print(color)
    saveColor(color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not using_pi:
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()
else:
    neopixels.fill( (0, 0, 0))

storage_file.flush()
storage_file.close()
