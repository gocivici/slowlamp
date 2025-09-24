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

4. Run your code!

Compatibility Note:
HD108 uses the same SPI interface as APA102/DotStar LEDs with the same clock and data pins (GPIO 10/11). Most online APA102 wiring diagrams will work for HD108 connections, just note that HD108 uses different data formatting in software.

