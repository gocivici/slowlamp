# https://learnopencv.com/exposure-fusion-using-opencv-cpp-python/
import cv2
import numpy as np
import glob

import colour

# Convert sRGB to Oklab
srgb_color = [0.5, 0.2, 0.8]  # Example sRGB color
oklab_color = colour.convert(srgb_color, "sRGB", "Oklab")
print(oklab_color)

# Convert Oklab to sRGB
oklab_color_back = [0.4, 0.1, -0.3] # Example Oklab color
srgb_color_back = colour.convert(oklab_color_back, "Oklab", "sRGB")
print(srgb_color_back)

# exit()

def srgb_to_oklab(img):
    h, w = img.shape[:2]
    oklab_img = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            srgb_color = img[y, x]
            oklab_color = colour.convert(srgb_color, "sRGB", "Oklab")
            oklab_img[y, x] = oklab_color
    return oklab_img

def get_diff_img(previous_img, current_img):
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
    return fgimg

def color_detection_lab(diff_img):
    image_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB) #convert to RGB image
    # image_rgb = diff_img
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    k = 5 # number of clusters
    #Apply the K-Means clustering
    DominantColors = np.array([[255, 255, 255, 0]] * k)
    label_counts_sorted = [1]*k

    if len(imgNoB.flatten()) <= 10: #needs to be greater than K
        print("not enough difference")
        return DominantColors, label_counts_sorted
    
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
    except:
        pass 

    return DominantColors, label_counts_sorted

def color_detection_oklab(diff_img):
    print("color_detection_oklab")
    image_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB) #convert to RGB image
    # image_rgb = diff_img
    # cv2.imshow("image-rgb", image_rgb)
    # print(f"original : {image_rgb}")
    pixels = image_rgb.reshape(-1, 3)
    # print(f"reshaped : {pixels}")
    imgNoB=pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    # print(f"background removed : {imgNoB}")

    k = 5 # number of clusters
    #Apply the K-Means clustering
    DominantColors = np.array([[255, 255, 255, 0]] * k)
    label_counts_sorted = [1]*k

    if len(imgNoB.flatten()) <= 10: #needs to be greater than K
        print("not enough difference")
        return DominantColors, label_counts_sorted
    
    imgNoB = imgNoB[np.newaxis, :, :]
    # if color_mode == "lab":
    # print("lab mode")
    print(f"The shape of pixle_values is : {imgNoB.shape}")

    imgNoB = srgb_to_oklab(imgNoB)

    imgNoB = np.float32(imgNoB) # convert to float. cv2.kmeans() requires to be in float 

    #Display the shape of pixel_values 
    
    #Define the stopping creteria and number of clusters
    # stop after 10 iterations or when EPS(change in cluster centers) is less than 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    
    
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
            rgb_color = colour.convert(color, "Oklab", "sRGB")
            DominantColors[idx] = list(rgb_color) + [alpha]  
            print(idx, color, rgb_color)
    except:
        raise 

    return DominantColors, label_counts_sorted

def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

def display_clusters(DominantColors, counts, diff_img):
    # display some stuff
    canvas = np.ones((500, 300, 3)).astype(np.uint8)
    # if canvas_color is not None:
    #     canvas[:, :, 0] = canvas_color[2]
    #     canvas[:, :, 1] = canvas_color[1]
    #     canvas[:, :, 2] = canvas_color[0]
    diff_preview = resize_to_max(diff_img, 300)
    canvas[0:diff_preview.shape[0], 0:diff_preview.shape[1]] = diff_preview

    feed_preview_y = diff_preview.shape[0]
    feed_preview_x = 0
    color_swatch_size = 40
    text_color = (127, 127, 127)
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 0.5  # Font scale (size)
    thickness = 1  # Thickness of the text

    for i, (count, center) in enumerate(zip(counts, DominantColors)):
        print("DominantColors (rgb) ", i, center)
        bgr_color = [center[2], center[1], center[0]]
        y_min = feed_preview_y + i*color_swatch_size
        canvas[y_min:y_min+color_swatch_size, feed_preview_x:feed_preview_x+color_swatch_size] = bgr_color
        canvas = cv2.putText(canvas, f'count {count}', (feed_preview_x+color_swatch_size, y_min+10), font, font_scale, text_color, thickness)

    return canvas

def compare_pair(path1, path2):

    # path3 = "C:/work/slow_lamp/september_coords/curim_20250905_130815.png"

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)# Create a GrayworldWB object
    # pip install opencv-contrib-python
    # wb = cv2.xphoto.createGrayworldWB()
    # img1 = wb.balanceWhite(img1)
    # img2 = wb.balanceWhite(img2)
    img1 = resize_to_max(img1, 800)
    img2 = resize_to_max(img2, 800)
    
    #----------- without stacking
    diff_plain = get_diff_img(img1, img2) #bgr
    colors_plain, counts_plain = color_detection_oklab(diff_plain)
    canvas_plain = display_clusters(colors_plain, counts_plain, diff_plain)

    # canvas = np.concatenate([canvas_plain, canvas_stacked], axis=1)
    # cv2.imshow("canvas", canvas_plain)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return colors_plain

image_paths = glob.glob("C:/work/slow_lamp/day_sept/curim_*.png")
image_paths = sorted(image_paths)
plain_history = []

for i in range(len(image_paths)-1):
    path1 = image_paths[i]
    path2 = image_paths[i+1]
    colors_plain = compare_pair(path1, path2)
    plain_history.append(colors_plain)
    # stacked_history.append(colors_stacked)

color_swatch_size = 40
aggregated = np.ones((color_swatch_size*5, color_swatch_size*(len(image_paths)-1), 3)).astype(np.uint8)

for i, (colors_plain) in enumerate(plain_history):
    for j, (center_plain) in enumerate(colors_plain):
        bgr_color = [center_plain[2], center_plain[1], center_plain[0]]
        y_min = color_swatch_size*j
        x_min = i*color_swatch_size
        aggregated[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size] = bgr_color
        # bgr_color = [center_stacked[2], center_stacked[1], center_stacked[0]]
        # y_min = color_swatch_size*5+color_swatch_size*j
        # x_min = i*color_swatch_size
        # aggregated[y_min:y_min+color_swatch_size, x_min:x_min+color_swatch_size] = bgr_color

cv2.imshow("aggregated", aggregated)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("C:/work/slow_lamp/day_sept/aggregated_oklab.png", aggregated)