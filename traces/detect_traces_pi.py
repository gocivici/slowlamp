import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import copy
import color_functions
from colormath.color_diff import delta_e_cie2000 #numpy error asscalar 
                # | ∆E$_{00}$ Value | Perceptual Difference                        |
                # | --------------- | -------------------------------------------- |
                # | 0–1             | **Not perceptible** by human eyes            |
                # | 1–2             | **Perceptible through close observation**    |
                # | 2–10            | **Noticeable at a glance**                   |
                # | 11–49           | **Colors are more different than similar**   |
                # | 50+             | **Perceived as completely different colors** |

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment

from picamera2 import Picamera2
# picamera2 needs numpy 1 seems like
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

canvas_size = 500
feed_preview_size = 100
color_swatch_size = 50

max_num_colors = 5
color_max_distance = 10

trace_length_min = 2
trace_length_max = 4
max_frame_number = 6


def rgb_to_lab(rgb):
    rgb_scaled = sRGBColor(*(rgb), is_upscaled=True) #expects 0~255
    lab_color = convert_color(rgb_scaled, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)


def lab_to_rgb(lab):
    # Create LabColor object
    lab_color = LabColor(*lab)
    # Convert to sRGB
    rgb_color = convert_color(lab_color, sRGBColor)
    
    # Convert to 0-255 integers (clip to valid range)
    rgb_clipped = np.clip([rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b], 0, 1)
    return (rgb_clipped * 255).astype(int)

def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

def extract_colors_kmeans(image, num_colors, resize=True, resize_max=200, color_mode="bgr"):

    if resize:
        image = resize_to_max(image, resize_max)

    if color_mode == "lab":
        # print("lab mode")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    img_np = copy.copy(image)
    pixels = img_np.reshape(-1, 3)

    # Fit KMeans to pixel data
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # Get dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    if color_mode == "lab":
        rgb_colors = cv2.cvtColor(np.array([colors]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
        return rgb_colors
    return colors

def extract_colors_gmm(image, max_colors, resize=True, resize_max=200, color_mode="bgr"):
    if resize:
        image = resize_to_max(image, resize_max)
    
    if color_mode == "lab":
        # print("lab mode")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # print(image[0, 0])

    img_np = copy.copy(image)
    pixels = img_np.reshape(-1, 3)

    lowest_bic = np.infty
    best_gmm = None
    for n_components in range(1, max_colors + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
        gmm.fit(pixels)
        bic = gmm.bic(pixels)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

      # Assign clusters
    labels = best_gmm.predict(pixels)

    # Count number of pixels per cluster
    counts = np.bincount(labels)
    
    # Get cluster centers (colors)
    colors = best_gmm.means_.astype(int)

    # Rank colors by frequency
    # ranked = sorted(zip(counts, colors), reverse=True)
    if color_mode == "lab":
        rgb_colors = cv2.cvtColor(np.array([colors]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
        return rgb_colors
    return colors

class ColorTrack:
    def __init__(self, color):
        self.color = np.array(color)
        self.age = 0
        self.missed = 0

    def update(self, new_color, alpha=0.3):
        self.color = (1 - alpha) * self.color + alpha * np.array(new_color)
        self.missed = 0
        self.age += 1

# Initialize the camera module
# cap = cv2.VideoCapture(0)

# Check if the camera is initialized correctly
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

tracks = [] #lab
canvas_color = None

# Display the camera feed
while True:
    # ret, frame = cap.read()
    frame = picam2.capture_array()
    # cv2.imshow('frame', frame)
    new_colors = extract_colors_kmeans(frame, max_num_colors, color_mode="lab") #rgb
    # print(new_colors)
    #if len(tracks) == 0:
        #tracks.append(dominant_colors)
    #    continue
    
    n, m = len(tracks), len(new_colors)
    cost_matrix = np.full((n, m), np.inf)

    # Build cost matrix
    for i, track in enumerate(tracks):
        for j, color in enumerate(new_colors):
            track_lab_color = rgb_to_lab(track.color)
            new_lab_color = rgb_to_lab(color)
            dist = color_functions.ciede2000(track_lab_color, new_lab_color)
            # print(dist)
            cost_matrix[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_tracks = set()
    assigned_colors = set()
    
    # Update matched tracks
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < color_max_distance:
            tracks[i].update(new_colors[j])
            assigned_tracks.add(i)
            assigned_colors.add(j)

    # Age and remove unmatched tracks
    new_tracks = []
    for i, track in enumerate(tracks):
        if i not in assigned_tracks:
            track.missed += 1
        if track.missed <= max_frame_number: # max_missed:
            new_tracks.append(track)
        if track.age >= trace_length_min and track.age <= trace_length_max and track.missed == trace_length_min:
            canvas_color = track.color #lab

    # Add new tracks for unmatched colors
    for j, color in enumerate(new_colors):
        if j not in assigned_colors:
            new_tracks.append(ColorTrack(color))
    
    tracks = new_tracks

    canvas = np.ones((500, 500, 3)).astype(np.uint8)
    if canvas_color is not None:
        canvas_rgb_color = canvas_color #lab_to_rgb(canvas_color)
        canvas[:, :, 0] = canvas_rgb_color[2]
        canvas[:, :, 1] = canvas_rgb_color[1]
        canvas[:, :, 2] = canvas_rgb_color[0]
    feed_preview = resize_to_max(frame, feed_preview_size)
    canvas[0:feed_preview.shape[0], 0:feed_preview.shape[1]] = feed_preview
    feed_preview_y = feed_preview.shape[0]
    for ci in range(len(new_colors)): #lab
        # start at feed_preview
        rgb_color = new_colors[ci]
        # print(lab_color)
        # rgb_color = lab_color #lab_to_rgb(lab_color)
        y_min = feed_preview_y + ci*color_swatch_size
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 0] = rgb_color[2]
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 1] = rgb_color[1]
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 2] = rgb_color[0]

    cv2.imshow('frame', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources
cap.release()
cv2.destroyAllWindows()

