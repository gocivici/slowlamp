# slowlamp
Hi, this is the slow lamp project. 

### notes
- source venvname/bin/activate
- pip install -U scikit-learn
- pip install colormath


- sudo apt install gh
- gh auth login

use picamera2 instead of opencv camera on raspberry pi 5

- sudo apt install python3-opencv or pip3 install opencv-python. 
- Make sure your system is updated with sudo apt update before installation.


picamera2 needs numpy 1 seems like: currently using 1.26 on pi

- https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi
- https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage
	- sudo ~/venvname/bin/pip install rpi_ws281x adafruit-circuitpython-neopixel


- sudo ~/slowlampvenv/bin/python3 detect_traces.py