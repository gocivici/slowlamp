from picamera2 import Picamera2
from pprintpp import pprint as pp
import numpy as np
from datetime import datetime
import time

day_length = 60 #min

#PiCamera setup
picam2 = Picamera2() #instantiates a picamera
modes = picam2.sensor_modes
mode = modes[1]
print('mode selected: ', mode)
camera_config = picam2.create_still_configuration(raw={'format': mode['unpacked']}, sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']})
picam2.configure(camera_config)
# Checking raw configuration
check = picam2.camera_configuration()['raw']
pp(check)
picam2.start()

# grab a RAW frame and save it as a np 16bit array.
# raw = picam2.capture_array("raw").view(np.uint16)

picam2.capture_file("test.dng", 'raw')
waitTime = 0

while True:
    print("Waiting seconds:",waitTime)
    if waitTime > 0:
        time.sleep(waitTime) 
    
    # Get the current time in seconds since the epoch
    start_time_seconds = time.time()

    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    exposure_time = [3000000, 2000000, 1000000, 500000, 250000, 125000, 62500]
    for exp in exposure_time: 
        picam2.set_controls({"ExposureTime": exp, "AnalogueGain": 1.0}) 
        picam2.start()
        file_name = f"frame_{filename_time}_{exp}.dng"
        picam2.capture_file(file_name, 'raw')

    time_elapsed = time.time() - start_time_seconds  
    waitTime = day_length*60 - time_elapsed
