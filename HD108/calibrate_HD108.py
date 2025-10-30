import cv2
import numpy as np
import time
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)                                            # Open SPI bus 0, device 0
spi.max_speed_hz = 2000000                               # Set speed to 10MHz: 32000000 maximum


using_pi = True
picam2 = None
if using_pi:
    from picamera2 import Picamera2
    # picamera2 needs numpy 1 seems like
    tuning = Picamera2.load_tuning_file("/home/slowlamp2/Documents/slowlamp/image_proc/test.json") 
    picam2 = Picamera2(tuning=tuning)
    picam2.preview_configuration.main.size = (1280,720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")

    picam2.set_controls({'AwbEnable': False})
    picam2.set_controls({'AeEnable': False})
    picam2.set_controls({'AnalogueGain': 7.5, 'ExposureTime': 15000})

    picam2.start()


pixel_colors = []
# for r in range(0, 255, 10):
# 	for g in range(0, 255, 10):
# 		for b in range(0, 255, 10):
# 			pixel_colors.append( (r, g, b))

for g in list(range(0, 255, 32))+[255]:
	for b in list(range(0, 255, 32))+[255]:
		for r in list(range(0, 255, 32))+[255]:
			pixel_colors.append( (r, g, b))

result_colors = []

canvas_size = 500
feed_preview_size = 100
color_swatch_size = 50

text_color = (127, 127, 127)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.5  # Font scale (size)
thickness = 1  # Thickness of the text


max_num_colors = 5
color_max_distance = 12

trace_length_min = 3
trace_length_max = 6
max_frame_number = 10


def resize_to_max(image, resize_max):

    h, w, = image.shape[:-1]
    if h>w:
        factor = resize_max/h
    else:
        factor = resize_max/w

    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    return image

def extract_center_color(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    crop = frame[cy-15:cy+15, cx-15:cx+15]
    avg_color = crop.mean(axis=(0, 1))  # BGR
    return avg_color[::-1]  # Return as RGB
    
def gamma_correct(x, gamma=2.4):
    return int((2**16-1)*(x/(2**16-1))**gamma)

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

test_index = 0

def send_hd108_colors(colors_16bit, global_brightness=4):
    
    data = []
    data.extend([0x00] * 8)  # Start frame: HD108 protocol requires 64-bit start frame before LED data
    
    for r16, g16, b16 in list(colors_16bit):
        r16, g16, b16 = map(int, (r16, g16, b16))
        
        # print(int(r16/255), int(g16/255), int(b16/255))
        if global_brightness <= 0:
            brightness = 0 #gives werid colors
        else:
            brightness = (1 << 15) | (global_brightness << 10) | (global_brightness << 5) | global_brightness
        
        data.extend([
            (brightness >> 8) & 0xFF, brightness & 0xFF,             # Brightness
            (r16 >> 8) & 0xFF, r16 & 0xFF,             # Blue 16-bit
            (g16 >> 8) & 0xFF, g16 & 0xFF,             # Green 16-bit
            (b16 >> 8) & 0xFF, b16 & 0xFF              # Red 16-bit
        ])
        
    num_end = 2 * (len(colors_16bit) + 1) 
    data.extend([0xFF] * num_end)  # End frame
    spi.writebytes(data)
    

# Display the camera feed
while test_index < len(pixel_colors):
    test_color = pixel_colors[test_index]
    upscaled = (test_color[0]*2**8, test_color[1]*2**7, test_color[2]*2**7) #875 for uncorrected
    # upscaled = (gamma_correct(upscaled[0]), gamma_correct(upscaled[1]), gamma_correct(upscaled[2]))
    if using_pi:
        send_hd108_colors([upscaled]*11)
        send_hd108_colors([upscaled]*11)
    
    time.sleep(0.6)

    test_index += 1

    if using_pi:
        frame = picam2.capture_array()
    else:
        ret, frame = cap.read()

    # cv2.imshow('frame', frame)
    recevied_color = extract_center_color(frame) #rgb
    result_colors.append(recevied_color)

    canvas = np.ones((500, 500, 3)).astype(np.uint8)
    feed_preview = resize_to_max(frame, feed_preview_size)
    canvas[0:feed_preview.shape[0], 0:feed_preview.shape[1]] = feed_preview
    feed_preview_y = feed_preview.shape[0]
    feed_preview_x = feed_preview.shape[1]
    canvas[feed_preview.shape[0]:100, 0:feed_preview.shape[1], :] = np.array(test_color)[::-1]
    canvas = cv2.putText(canvas, f'pixel (rgbw): ({test_color[0]}, {test_color[1]}, {test_color[2]})', (feed_preview_x, 20), font, font_scale, text_color, thickness)
    canvas = cv2.putText(canvas, f'camera (rgb): ({int(recevied_color[0])}, {int(recevied_color[1])}, {int(recevied_color[2])})', (feed_preview_x, 50), font, font_scale, text_color, thickness)
        

    cv2.imshow('frame', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if not using_pi:
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()
else:
    colors_16bit = [(0, 0, 0)]*11
    send_hd108_colors(colors_16bit, global_brightness = 1)
    send_hd108_colors(colors_16bit, global_brightness = 1)
    time.sleep(1)
    spi.close()

np.savez("pairings-32-white_HD108_877_with_light.npz", pixel_colors = pixel_colors, result_colors = result_colors)
