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

from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
from scipy.optimize import linear_sum_assignment
import time 
from datetime import datetime

using_pi = True
pixel_mode = "daily" # "reactive" or "daily"
traces_storing_mode = "complementary" # "single", "complementary", or "neighbor"
display_matrix_mode = "gradient"

day_length = 5 #minutes
filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_file = open(f"{filename_time}_storage.txt", "w")

start_time = time.time()

picam2 = None
if using_pi:
    import board
    import neopixel
    pixels = neopixel.NeoPixel(board.D12, 24, brightness=0.5, pixel_order = neopixel.GRBW)

    from picamera2 import Picamera2
    # picamera2 needs numpy 1 seems like
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280,720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    
    picam2.set_controls({'AnalogueGain': 12.0, 'ExposureTime': 17000})
    picam2.start()

canvas_size = 500
feed_preview_size = 100
color_swatch_size = 50

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text


max_num_colors = 5
color_max_distance = 12

trace_length_min = 2
trace_length_max = 5
max_frame_number = 12


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

def rgb_to_lch(rgb):
    rgb_scaled = sRGBColor(*[v / 255 for v in rgb], is_upscaled=False)
    lab = convert_color(rgb_scaled, LabColor)
    lch = convert_color(lab, LCHabColor)
    # test = LCHabColor()
    return (lch.lch_l, lch.lch_c, lch.lch_h)

def hue_distance(h1, h2):
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)

def complementarity_score(lch1, lch2):
    h1 = lch1[2]
    h2 = lch2[2]

    hue_score = 1 - abs(hue_distance(h1, h2) - 180) / 180
    chroma_score = 1 - abs(lch1[1] - lch2[1])/100
    luminance_score = 1 - abs(lch1[0] - lch2[0])/100
    return hue_score + 0.1*chroma_score + 0.1*luminance_score

def get_chroma(rgb): 
    # how saturated or vivid one color is 
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
    # labels = best_gmm.predict(pixels)

    # Count number of pixels per cluster
    # counts = np.bincount(labels)
    
    # Get cluster centers (colors)
    colors = best_gmm.means_.astype(int)

    # Rank colors by frequency
    # ranked = sorted(zip(counts, colors), reverse=True)
    if color_mode == "lab":
        rgb_colors = cv2.cvtColor(np.array([colors]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
        return rgb_colors
    return colors

def choose_trace(traces, mode = "chroma_single"):
    chosen_id = 0
    if mode == "chroma_single":
        max_chromma = 0
        for i, tr in enumerate(traces):
            main_color = tr.main_color
            chroma = get_chroma(main_color)
            if chroma >= max_chromma:
                max_chromma = chroma
                chosen_id = i
    return traces[chosen_id]


class ColorTrack:
    def __init__(self, color):
        self.color = np.array(color)
        self.age = 0
        self.missed = 0

    def update(self, new_color, alpha=0.3):
        self.color = (1 - alpha) * self.color + alpha * np.array(new_color)
        self.missed = 0
        self.age += 1

    def print_swatch(self):
        return self.color, str(self.age), "-"*self.missed
    

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
day_traces = []
stored_traces = []

# Display the camera feed
while True:

    if using_pi:
        frame = picam2.capture_array()
    else:
        ret, frame = cap.read()

    time.sleep(0.4)

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

    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # match color to closest existing track, not a global optimization 
    if len(cost_matrix) > 0:
        closest_track_ids = np.argmin(cost_matrix, axis=0)
    else:
        closest_track_ids = []

    assigned_tracks = set()
    assigned_colors = set()
    
    # # Update matched tracks (after linear sum assignment)
    # for i, j in zip(row_ind, col_ind):
    #     if cost_matrix[i, j] < color_max_distance:
    #         tracks[i].update(new_colors[j])
    #         assigned_tracks.add(i)
    #         assigned_colors.add(j)

    for ni, ti in enumerate(closest_track_ids):
        if cost_matrix[ti, ni] < color_max_distance:
            tracks[ti].update(new_colors[ni])
            assigned_tracks.add(ti)
            assigned_colors.add(ni)

    # Age and remove unmatched tracks
    new_tracks = []
    new_candidate_detected = False
    for i, track in enumerate(tracks):
        if i not in assigned_tracks:
            track.missed += 1
        if track.missed <= max_frame_number: # max_missed:
            new_tracks.append(track)
        if track.age >= trace_length_min and track.age <= trace_length_max and track.missed == trace_length_min:
            canvas_color = track.color #rgb
            new_candidate_detected = True
            if memory_color is None: 
                memory_color = track.color
            
            if traces_storing_mode == "single":
                trace = Trace(track.color, traces_storing_mode=traces_storing_mode)
                day_traces.append(trace)
                if pixel_mode == "reactive":
                    storage_file.writelines([trace.print_trace()])
                    stored_traces.append(trace)

            if using_pi and pixel_mode == "reactive":
                pixels[0] = (track.color[0], track.color[1], track.color[2])

    # Add new tracks for unmatched colors
    for j, color in enumerate(new_colors):
        if j not in assigned_colors:
            new_tracks.append(ColorTrack(color))

    if traces_storing_mode != "single" and new_candidate_detected:
        if traces_storing_mode == "complementary":
            candidate_lch = rgb_to_lch(canvas_color)

            short_list = []
            for i, track in enumerate(new_tracks):
                rgb_color, _, _ = track.print_swatch()
                track_lch = rgb_to_lch(rgb_color)
                if track_lch[1] - candidate_lch[1] < 30:
                    short_list.append(rgb_color)

            comp_color = (0, 0, 0)
            max_comp_score = 0
            if len(short_list) >= 1:
                for i, track_rgb in enumerate(short_list):
                    track_lch = rgb_to_lch(track_rgb)
                    comp_score = complementarity_score(candidate_lch, track_lch)
                    if comp_score >= max_comp_score:
                        max_comp_score = comp_score
                        comp_color = track_rgb
            else:
                print("no color with similar L or C in view - choose the other most satuared")
                max_chromma = 0
                for i, track in enumerate(new_tracks):
                    rgb_color, _, _ = track.print_swatch()
                    if rgb_color == canvas_color:
                        continue
                    chroma = get_chroma(rgb_color)
                    if chroma >= max_chromma:
                        max_chromma = chroma
                        comp_color = rgb_color

            trace = Trace(canvas_color, traces_storing_mode=traces_storing_mode, supplemental_colors=[comp_color])
            day_traces.append(trace)

    
    tracks = new_tracks

    # diagonstic visualization 
    canvas = np.ones((700, 700, 3)).astype(np.uint8)
    if canvas_color is not None:
        canvas_rgb_color = canvas_color #lab_to_rgb(canvas_color)
        canvas[:, :, 0] = canvas_rgb_color[2]
        canvas[:, :, 1] = canvas_rgb_color[1]
        canvas[:, :, 2] = canvas_rgb_color[0]
    feed_preview = resize_to_max(frame, feed_preview_size)
    canvas[0:feed_preview.shape[0], 0:feed_preview.shape[1]] = feed_preview
    feed_preview_y = feed_preview.shape[0]
    feed_preview_x = feed_preview.shape[1]
    for ci in range(len(new_colors)): #lab
        # start at feed_preview
        rgb_color = new_colors[ci]
        # print(lab_color)
        # rgb_color = lab_color #lab_to_rgb(lab_color)
        y_min = feed_preview_y + ci*color_swatch_size
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 0] = rgb_color[2]
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 1] = rgb_color[1]
        canvas[y_min:y_min+color_swatch_size, 0:color_swatch_size, 2] = rgb_color[0]

    for i, track in enumerate(tracks):
        rgb_color, age_str, missed_str = track.print_swatch()
        bgr_color = [rgb_color[2], rgb_color[1], rgb_color[0]]
        y_min = feed_preview_y + i*color_swatch_size
        canvas[y_min:y_min+color_swatch_size, feed_preview_x:feed_preview_x+color_swatch_size] = bgr_color
        canvas = cv2.putText(canvas, f'age{age_str}', (feed_preview_x+color_swatch_size, y_min+10), font, font_scale, text_color, thickness)
        canvas = cv2.putText(canvas, f'missed{missed_str}', (feed_preview_x+color_swatch_size, y_min+25), font, font_scale, text_color, thickness)

    for i, trace in enumerate(stored_traces):
        swatch = trace.paint_trace()
        sh, sw = swatch.shape[:-1]
        y_min = i*color_swatch_size
        x_min = feed_preview_x+200
        if y_min+sh <= len(canvas):
            canvas[y_min: y_min+sh, x_min:x_min+sw] = swatch
        else:
            # clear "stored" traces used in diagonstic visualization
            stored_traces = []

    cv2.imshow('frame', canvas)
    
    # update pixel at the end of the day
    if pixel_mode == "daily":
        elapsed = time.time() - start_time
        mins, secs = divmod(elapsed, 60)
        # print(f"Elapsed time: {int(mins):02}:{int(secs):02}")
        if mins >= day_length:
            # decide memory color
            if len(day_traces) > 0:
                stored_trace = choose_trace(day_traces)
                storage_file.writelines([stored_trace.print_trace()])
                stored_traces.append(stored_trace)
                day_traces = []
                if using_pi: 
                    memory_color = (int(stored_trace.main_color[0]), int(stored_trace.main_color[1]), int(stored_trace.main_color[2]), 0)
                    pixels[0] = memory_color
                    if stored_trace.traces_storing_mode != "single":
                        sup_color = (int(stored_trace.supplemental_colors[0][0]), int(stored_trace.supplemental_colors[0][1]), int(stored_trace.supplemental_colors[0][2]), 0)
                        if display_matrix_mode == "checkered":
                            for strip in range(3):
                                for i in range(8):
                                    if i%2 == 1:
                                        pixels[strip*8+i] = sup_color
                                    else:
                                        pixels[strip*8+i] = memory_color
                        else:
                            for strip in range(3):
                                for i in range(8):
                                    if strip % 2 == 1:
                                        pixels[strip*8+i] = linear_gradient(sup_color, memory_color, i/8)
                                    else:
                                        pixels[strip*8+i] = linear_gradient(memory_color, sup_color, i/8)
                                
                            
            start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not using_pi:
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()
else:
    pixels.fill( (0, 0, 0))

storage_file.flush()
storage_file.close()
