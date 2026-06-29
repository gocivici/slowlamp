import smbus
import time
import threading

# --- Hardware Setup ---
AS5600_ADDRESS = 0x36
RAW_ANGLE_HI = 0x0C
RAW_ANGLE_LO = 0x0D

TEETH_SENSOR_GEAR = 17
TEETH_BIG_GEAR = 43
GEAR_RATIO = TEETH_SENSOR_GEAR / TEETH_BIG_GEAR

# --- Mode Setup ---
MODE_TOLERANCE = 20.0
MODES = {
    225.0: "day",
    315.0: "month",
    45.0: "year",
    135.0: "all_time",
}

MODE_TO_IMAGE = {
    "day": "view_24h.png",
    "month": "view_30d.png",
    "year": "view_365d.png",
    "all_time": "view_all_time.png",
}

# --- Interaction Setup ---
MIN_MOVEMENT_SPEED = 10
SCROLL_TRIGGER_DEGREES = 360.0
PAUSE_TIMEOUT = 0.3
SCROLL_TIMEOUT = 1.0

# --- Global State ---
previous_raw = None
accumulated_raw = 0
current_mode = "day"
_lock = threading.Lock()

try:
    bus = smbus.SMBus(1)
except Exception as e:
    bus = None
    print(f"Error initializing I2C bus: {e}")


def get_angle():
    try:
        high = bus.read_byte_data(AS5600_ADDRESS, RAW_ANGLE_HI)
        low = bus.read_byte_data(AS5600_ADDRESS, RAW_ANGLE_LO)
        return (high << 8) | low
    except Exception:
        return None


def calibrate():
    global previous_raw, accumulated_raw
    raw = None
    while raw is None:
        raw = get_angle()
        time.sleep(0.05)
    previous_raw = raw
    accumulated_raw = 0


def determine_mode(current_degrees):
    for target_angle, mode_name in MODES.items():
        diff = abs(current_degrees - target_angle)
        if diff > 180.0:
            diff = 360.0 - diff
        if diff <= MODE_TOLERANCE:
            return mode_name
    return None


def get_current_mode():
    with _lock:
        return current_mode


def get_current_image():
    with _lock:
        return MODE_TO_IMAGE.get(current_mode, "view_24h.png")


def interaction_loop():
    global previous_raw, accumulated_raw, current_mode

    if bus is None:
        return

    calibrate()

    in_scroll_mode = False
    continuous_distance = 0
    last_move_time = time.time()

    while True:
        current_raw = get_angle()
        current_time = time.time()

        if current_raw is not None:
            delta = current_raw - previous_raw

            if delta > 2048:
                delta -= 4096
            elif delta < -2048:
                delta += 4096

            accumulated_raw += delta
            previous_raw = current_raw

            if abs(delta) > MIN_MOVEMENT_SPEED:
                continuous_distance += delta
                last_move_time = current_time
            else:
                if (current_time - last_move_time) > PAUSE_TIMEOUT:
                    continuous_distance = 0

            run_up_degrees = abs((continuous_distance / 4096.0) * 360.0 * GEAR_RATIO)

            if not in_scroll_mode:
                if run_up_degrees >= SCROLL_TRIGGER_DEGREES:
                    in_scroll_mode = True
            else:
                if (current_time - last_move_time) > SCROLL_TIMEOUT:
                    in_scroll_mode = False
                    continuous_distance = 0

            sensor_degrees = (accumulated_raw / 4096.0) * 360.0
            big_gear_normalized = (sensor_degrees * GEAR_RATIO) % 360.0

            if in_scroll_mode:
                resolved_mode = "scroll"
            else:
                resolved_mode = determine_mode(big_gear_normalized)

            if resolved_mode and resolved_mode != "scroll":
                with _lock:
                    current_mode = resolved_mode

        time.sleep(0.1)


def start():
    thread = threading.Thread(target=interaction_loop, daemon=True)
    thread.start()
    return thread
