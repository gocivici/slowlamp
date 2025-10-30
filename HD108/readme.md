## how to run HD108
1. Enable SPI on Raspberry Pi:

    sudo raspi-config

    Navigate: Interface Options → SPI → Enable → Reboot
   

2. Make the connections:

    Pi GPIO 10 (MOSI) → HD108 SDI (data)

    Pi GPIO 11 (SCLK) → HD108 CKI (clock)

    Pi GND → HD108 GND

    5V power supply (+) → HD108 VCC

    5V power supply (-) → HD108 GND (same connection as Pi GND)

    For 5 LEDs: A small 5V/1A power supply works fine

    Ensure common ground: Connect Pi GND, power supply GND, and LED strip GND together
   

3. Install Python SPI library:

    sudo apt update
   
    sudo apt install python3-spidev
   

5. Run your code!

    Compatibility Note:
   
    HD108 uses the same SPI interface as APA102/DotStar LEDs with the same clock and data pins (GPIO 10/11).
   
    Most online APA102 wiring diagrams will work for HD108 connections, just note that HD108 uses different data formatting in code.

## code explanations 

- `arduino_nano_HD108/` contains the code to teset HD108 on an Arduino Nano
- `animation_X.py` animates the led strips to compare the smoothness
- `calibrate_HD108.py` loops through the rgb values between 0 and 255 at steps of 32 and display this on the HD108 strip while capturing the resulting color with the raspberry pi camera
- `compare_HD08_with_screen.py` loops through the rgb values between 0 and 255 at steps of 32 and display both the rgb directly on the HD108 and the corrected color value based on the ml model imported via `calibrate_HD108.py`
- `correct_color_HD108.py` contains the code to build the statistical model and a few visualization functions. 
- `other_plotting_and_exploration.py` is based on `correct_color_HD108.py` but plots the difference between diffusers. 
- the `.npz` files are the outputs I gathered from `calibrate_HD108.py` needed for `correct_color_HD108.py`. 

## calibration workflow

1. Change the output file name and location in `calibrate_HD108.py` and make sure to send proper data for the number of leds in `send_hd108_colors([upscaled]*>>>11<<<)`. Change how you want to scale it, apply gamma or not, etc. 

2. Load the output npz into `correct_color_HD108.py` and change the model filename to save as a pkl. Optional: Run to see the outputs and generate a pkl of the model. 

3. Run `compare_HD08_with_screen.py` with the correct import of model/npz in `correct_color_HD108.py` to observe the results. 