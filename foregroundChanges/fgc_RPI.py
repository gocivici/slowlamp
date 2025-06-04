import numpy as np
# import matplotlib.pyplot as plt
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46
import time
import math
import os
from PIL import Image
from picamera2 import Picamera2


#------------------------Camera Setup---------------------------------------
camera = Picamera2()
camera.resolution= (480,360)
camera.start(show_preview=False)

def capture_image():

    frame = camera.capture_array()
    # cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
    # if not cap.isOpened():
    #     raise Exception("Could not open webcam")
    # ret, frame = cap.read()
    # cap.release()
    # if not ret:
    #     raise Exception("Failed to capture image")
    return frame

def dominantColor():

    

    previous_img = capture_image()

    previous_img = cv2.resize(previous_img, (480, 360))

# cv2.imshow("web live",previous_img)


    print("Waiting 10 seconds")
    time.sleep(60)  # Wait 15 seconds


    # Capture the new image
    current_img = capture_image()

    current_img = cv2.resize(current_img, (480, 360))

#-------------------------Backgorund Substraction-----------------------------------
        # Compare with the previous image
    diff1=cv2.subtract(current_img,previous_img)
    diff2=cv2.subtract(previous_img,current_img)
    diff = diff1+diff2

        # Optional: Show difference
    # cv2.imshow("previous", previous_img)
    # cv2.imshow("current", current_img)
    # cv2.waitKey(1)  # Refresh the window

        # Update previous image for next iteration
    

    #adjustable threshold value original value =13
    diff[abs(diff)<50]=0

    #create mask based on threshold
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray[np.abs(gray) < 5] = 0
    fgmask = gray.astype(np.uint8)



    # morphological closing operation using ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


    #use the masks to extract the relevant parts from the image
    fgimg = cv2.bitwise_and(current_img,current_img,mask = morph)



    #-------------------------Color Detection-----------------------------------

    # image_rgb = cv2.cvtColor(fgimg,cv2.COLOR_BGR2RGB) #convert to RGB image
    image_rgb = fgimg
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    r = imgNoB[:, 0]
    g = imgNoB[:, 1]
    b = imgNoB[:, 2]

    colors = imgNoB / 255.0




    imgNoB = np.float32(imgNoB) # convert to float. cv2.kmeans() requires to be in float 

    #Display the shape of pixel_values 
    print(f"The shape of pixle_values is : {imgNoB.shape}")

    #Define the stopping creteria and number of clusters
    # stop after 10 iterations or when EPS(change in cluster centers) is less than 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    k = 3 # number of clusters

    #Apply the K-Means clustering
    DominantColors = np.array([[0, 0, 0, 0]] * 5)
    try:
        ret, label, center = cv2.kmeans(imgNoB, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        label= np.reshape(label, label.size)
        label_counts = np.bincount(label)

        sorted_indices = np.argsort(-label_counts)

        # Sort center and label_counts accordingly
        center_sorted = center[sorted_indices]
        label_counts_sorted = label_counts[sorted_indices]

        print(enumerate(center))
        for idx, color in enumerate(center_sorted):
            # plt.subplot(1, k, idx+1)
            # plt.axis('off')
            # plt.imshow([[color / 255]])  # Normalize RGB values to range [0,1]
            # plt.title(f'%{(100*label_counts_sorted[idx]/imgNoB.shape[0]):.2f}')
            alpha = 255 * label_counts_sorted[idx] / imgNoB.shape[0]
            DominantColors[idx] = list(color) + [alpha]  
            print(label_counts)
        print(DominantColors[0])
    except:
        DominantColors[0]=[255, 255, 255, 0]
    previous_img = current_img
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
    columns = 7
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





while True:
    color = dominantColor()
    print(color)
    saveColor(color)



