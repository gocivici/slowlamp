---
title: BOM
layout: default
nav_order: 2
---

# Hardware Bill of Materials (BOM)

The following components are required for a minimal DIY slow lamp build. Please note the specific cable and magnet requirements in the accessories section below.

| Component | Description & Specifications | Qty |
|:---|:---|:---:|
| **Raspberry Pi Zero 2 W** | Quad-core 64-bit ARM Cortex-A53, 512MB RAM, integrated Wi-Fi and Bluetooth. | 1 |
| **Waveshare Round LCD Display** | SPI/I2C round display module. *(Note: Waveshare's standard round screen is typically 1.28" at 240x240px resolution. Verify your exact model).* | 1 |
| **AS5600 Magnetic Encoder Module** | 12-bit absolute magnetic rotary encoder for precise position tracking (I2C interface). | 1 |
| **Raspberry Pi Camera Module v2.0** | 8-megapixel Sony IMX219 sensor (Standard or NoIR depending on lighting needs). | 1 |

## Recommended Accessories & Wiring

* **Diametric Magnet:** A standard magnet will not work; the AS5600 requires a diametrically magnetized magnet to properly read rotations.
* **MicroSD Card:** 16GB or larger (Class 10 recommended) for the Raspberry Pi
* **Power Supply:** 5V 2.5A Micro-USB power supply for the Pi Zero 2 W.
* **Jumper cables & Headers:** Jumper wires for connecting the I2C/SPI pins from the Pi to the encoder and display.

## Alternative Display Options

Here are a few tested alternatives that support the Raspberry Pi Zero 2 W.

* **Pimoroni HyperPixel 2.1 Round (2.1" | 480x480px):**
* **Waveshare 4-inch HDMI Round Touch Display (4.0" | 720x720px):**
* **Waveshare 1.69" or 1.9" IPS LCDs (Rounded Rectangles):**