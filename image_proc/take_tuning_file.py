#!/usr/bin/python3
import time

from picamera2 import Picamera2, Preview

# Here we load up the tuning for the HQ cam and alter the default exposure profile.
# For more information on what can be changed, see chapter 5 in
# https://datasheets.raspberrypi.com/camera/raspberry-pi-camera-guide.pdf

tuning = Picamera2.load_tuning_file("/home/slowlamp2/Documents/slowlamp/image_proc/imx477.json") #imx477
algo = Picamera2.find_tuning_algo(tuning, "rpi.agc")
if "channels" in algo:
    algo["channels"][0]["exposure_modes"]["normal"] = {"shutter": [100, 66666], "gain": [1.0, 8.0]}
else:
    algo["exposure_modes"]["normal"] = {"shutter": [100, 66666], "gain": [1.0, 8.0]}

picam2 = Picamera2(tuning=tuning)
picam2.configure(picam2.create_preview_configuration())
picam2.start_preview(Preview.QTGL)
picam2.start()
picam2.capture_file("original.jpg", "main")
time.sleep(2)
